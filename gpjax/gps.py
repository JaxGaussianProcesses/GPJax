import typing as tp
from abc import abstractmethod, abstractproperty

import distrax as dx
import jax.numpy as jnp
from chex import dataclass
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

from .config import get_defaults
from .kernels import Kernel, cross_covariance, gram
from .likelihoods import (
    AbstractLikelihood,
    Gaussian,
    NonConjugateLikelihoods,
    NonConjugateLikelihoodType,
    Conjugate,
    NonConjugate,
)
from .mean_functions import AbstractMeanFunction, Zero
from .parameters import copy_dict_structure, evaluate_priors, transform
from .types import Array, Dataset
from .utils import I, concat_dictionaries

DEFAULT_JITTER = get_defaults()["jitter"]


@dataclass
class AbstractGP:
    """Abstract Gaussian process object."""

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> dx.Distribution:
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(self, *args: tp.Any, **kwargs: tp.Any) -> dx.Distribution:
        """Compute the latent function's multivariate normal distribution."""
        raise NotImplementedError

    @abstractproperty
    def params(self) -> tp.Dict:
        """Initialise the GP's parameter set"""
        raise NotImplementedError


#######################
# GP Priors
#######################
@dataclass(repr=False)
class Prior(AbstractGP):
    """A Gaussian process prior object. The GP is parameterised by a mean and kernel function."""

    kernel: Kernel
    mean_function: tp.Optional[AbstractMeanFunction] = Zero()
    name: tp.Optional[str] = "Prior"
    jitter: tp.Optional[float] = DEFAULT_JITTER

    def __mul__(self, other: AbstractLikelihood):
        """The product of a prior and likelihood is proportional to the posterior distribution. By computing the product of a GP prior and a likelihood object, a posterior GP object will be returned.
        Args:
            other (Likelihood): The likelihood distribution of the observed dataset.
        Returns:
            Posterior: The relevant GP posterior for the given prior and likelihood. Special cases are accounted for where the model is conjugate.
        """
        return construct_posterior(prior=self, likelihood=other)

    def __rmul__(self, other: AbstractLikelihood):
        """Reimplement the multiplication operator to allow for order-invariant product of a likelihood and a prior i.e., likelihood * prior."""
        return self.__mul__(other)

    def predict(self, params: dict) -> tp.Callable[[Array], dx.Distribution]:
        """Compute the GP's prior mean and variance.
        Args:
            params (dict): The specific set of parameters for which the mean function should be defined for.
        Returns:
            tp.Callable[[Array], Array]: A mean function that accepts an input array for where the mean function should be evaluated at. The mean function's value at these points is then returned.
        """

        def predict_fn(test_inputs: Array) -> dx.Distribution:
            t = test_inputs
            μt = self.mean_function(t, params["mean_function"])
            Ktt = gram(self.kernel, t, params["kernel"])
            Ktt += I(t.shape[0]) * self.jitter

            return dx.MultivariateNormalFullCovariance(jnp.atleast_1d(μt.squeeze()), Ktt)

        return predict_fn

    @property
    def params(self) -> dict:
        """Initialise the GP prior's parameter set"""
        return {
            "kernel": self.kernel.params,
            "mean_function": self.mean_function.params,
        }


#######################
# GP Posteriors
#######################
@dataclass
class AbstractPosterior(AbstractGP):
    """The base GP posterior object conditioned on an observed dataset."""

    prior: Prior
    likelihood: AbstractLikelihood
    name: tp.Optional[str] = "GP Posterior"
    jitter: tp.Optional[float] = DEFAULT_JITTER

    @abstractmethod
    def predict(self, *args: tp.Any, **kwargs: tp.Any) -> dx.Distribution:
        """Predict the GP's output given the input."""
        raise NotImplementedError

    @property
    def params(self) -> dict:
        return concat_dictionaries(self.prior.params, {"likelihood": self.likelihood.params})


@dataclass
class ConjugatePosterior(AbstractPosterior):
    """Gaussian process posterior object for models where the likelihood is Gaussian."""

    prior: Prior
    likelihood: Gaussian
    name: tp.Optional[str] = "ConjugatePosterior"
    jitter: tp.Optional[float] = DEFAULT_JITTER

    def predict(self, train_data: Dataset, params: dict) -> tp.Callable[[Array], dx.Distribution]:
        """Conditional on a set of training data, compute the GP's posterior predictive distribution for a given set of parameters. The returned function can be evaluated at a set of test inputs to compute the corresponding predictive density.

        Args:
            train_data (Dataset): A `gpx.Dataset` object that contains the input and output data used for training dataset.
            params (dict): A dictionary of parameters that should be used to compute the posterior.

        Returns:
            tp.Callable[[Array], dx.Distribution]: A function that accepts an input array and returns the predictive distribution as a `distrax.MultivariateNormalFullCovariance`.
        """
        x, y, n = train_data.X, train_data.y, train_data.n

        obs_noise = params["likelihood"]["obs_noise"]
        μx = self.prior.mean_function(x, params["mean_function"])

        # Precompute covariance matrices
        Kxx = gram(self.prior.kernel, x, params["kernel"])
        Kxx += I(n) * self.jitter
        Lx = cho_factor(Kxx + I(n) * obs_noise, lower=True)

        weights = cho_solve(Lx, y - μx)

        def predict(test_inputs: Array) -> dx.Distribution:
            t = test_inputs

            # Compute the mean
            μt = self.prior.mean_function(t, params["mean_function"])
            Ktx = cross_covariance(self.prior.kernel, t, x, params["kernel"])
            mean = μt + jnp.dot(Ktx, weights)

            # Compute the covariance
            Ktt = gram(self.prior.kernel, t, params["kernel"])
            latent_values = cho_solve(Lx, Ktx.T)
            covariance = Ktt - jnp.dot(Ktx, latent_values)
            covariance += I(t.shape[0]) * self.jitter

            return dx.MultivariateNormalFullCovariance(jnp.atleast_1d(mean.squeeze()), covariance)

        return predict

    def marginal_log_likelihood(
        self,
        train_data: Dataset,
        transformations: tp.Dict,
        priors: dict = None,
        negative: bool = False,
    ) -> tp.Callable[[dict], Array]:
        """Compute the marginal log-likelihood function of the Gaussian process. The returned function can then be used for gradient based optimisation of the model's parameters or for model comparison. The implementation given here enables exact estimation of the Gaussian process' latent function values.

        Args:
            train_data (Dataset): The training dataset used to compute the marginal log-likelihood.
            transformations (tp.Dict): A dictionary of transformations that should be applied to the training dataset to unconstrain the parameters.
            priors (dict, optional): _description_. Optional argument that contains the priors placed on the model's parameters. Defaults to None.
            negative (bool, optional): Whether or not the returned function should be negative. For optimisation, the negative is useful as minimisation of the negative marginal log-likelihood is equivalent to maximisation of the marginal log-likelihood. Defaults to False.

        Returns:
            tp.Callable[[dict], Array]: A functional representation of the marginal log-likelihood that can be evaluated at a given parameter set.
        """
        x, y, n = train_data.X, train_data.y, train_data.n

        def mll(
            params: dict,
        ):
            params = transform(params=params, transform_map=transformations)

            obs_noise = params["likelihood"]["obs_noise"]
            μx = self.prior.mean_function(x, params["mean_function"])
            Kxx = gram(self.prior.kernel, x, params["kernel"])
            Kxx += I(n) * self.jitter
            Lx = jnp.linalg.cholesky(Kxx + I(n) * obs_noise)

            random_variable = dx.MultivariateNormalTri(μx.squeeze(), Lx)

            log_prior_density = evaluate_priors(params, priors)
            constant = jnp.array(-1.0) if negative else jnp.array(1.0)
            return constant * (random_variable.log_prob(y.squeeze()).mean() + log_prior_density)

        return mll


@dataclass
class NonConjugatePosterior(AbstractPosterior):
    """Generic Gaussian process posterior object for models where the likelihood is non-Gaussian."""

    prior: Prior
    likelihood: NonConjugateLikelihoodType
    name: tp.Optional[str] = "Non-Conjugate Posterior"
    jitter: tp.Optional[float] = DEFAULT_JITTER

    @property
    def params(self) -> dict:
        hyperparameters = concat_dictionaries(
            self.prior.params, {"likelihood": self.likelihood.params}
        )
        hyperparameters["latent"] = jnp.zeros(shape=(self.likelihood.num_datapoints, 1))
        return hyperparameters

    def predict(self, train_data: Dataset, params: dict) -> tp.Callable[[Array], dx.Distribution]:
        """Conditional on a set of training data, compute the GP's posterior predictive distribution for a given set of parameters. The returned function can be evaluated at a set of test inputs to compute the corresponding predictive density. Note, to gain predictions on the scale of the original data, the returned distribution will need to be transformed through the likelihood function's inverse link function.

        Args:
            train_data (Dataset): A `gpx.Dataset` object that contains the input and output data used for training dataset.
            params (dict): A dictionary of parameters that should be used to compute the posterior.

        Returns:
            tp.Callable[[Array], dx.Distribution]: A function that accepts an input array and returns the predictive distribution as a `distrax.MultivariateNormalFullCovariance`.
        """
        x, n = train_data.X, train_data.n

        Kxx = gram(self.prior.kernel, x, params["kernel"])
        Kxx += I(n) * self.jitter
        Lx = jnp.linalg.cholesky(Kxx)

        def predict_fn(test_inputs: Array) -> dx.Distribution:
            t = test_inputs
            Ktx = cross_covariance(self.prior.kernel, t, x, params["kernel"])
            Ktt = gram(self.prior.kernel, t, params["kernel"])
            μt = self.prior.mean_function(t, params["mean_function"])
            A = solve_triangular(Lx, Ktx.T, lower=True)
            latent_var = Ktt - jnp.sum(jnp.square(A), -2)
            latent_mean = μt + jnp.matmul(A.T, params["latent"])
            return dx.MultivariateNormalFullCovariance(
                jnp.atleast_1d(latent_mean.squeeze()), latent_var
            )

        return predict_fn

    def marginal_log_likelihood(
        self,
        train_data: Dataset,
        transformations: tp.Dict,
        priors: dict = None,
        negative: bool = False,
    ) -> tp.Callable[[dict], Array]:
        """Compute the marginal log-likelihood function of the Gaussian process. The returned function can then be used for gradient based optimisation of the model's parameters or for model comparison. The implementation given here is general and will work for any likelihood support by GPJax.

        Args:
            train_data (Dataset): The training dataset used to compute the marginal log-likelihood.
            transformations (tp.Dict): A dictionary of transformations that should be applied to the training dataset to unconstrain the parameters.
            priors (dict, optional): _description_. Optional argument that contains the priors placed on the model's parameters. Defaults to None.
            negative (bool, optional): Whether or not the returned function should be negative. For optimisation, the negative is useful as minimisation of the negative marginal log-likelihood is equivalent to maximisation of the marginal log-likelihood. Defaults to False.

        Returns:
            tp.Callable[[dict], Array]: A functional representation of the marginal log-likelihood that can be evaluated at a given parameter set.
        """
        x, y, n = train_data.X, train_data.y, train_data.n

        if not priors:
            priors = copy_dict_structure(self.params)
            priors["latent"] = dx.Normal(loc=0.0, scale=1.0)

        def mll(params: dict):
            params = transform(params=params, transform_map=transformations)
            Kxx = gram(self.prior.kernel, x, params["kernel"])
            Kxx += I(n) * self.jitter
            Lx = jnp.linalg.cholesky(Kxx)
            μx = self.prior.mean_function(x, params["mean_function"])
            fx = jnp.matmul(Lx, params["latent"]) + μx
            rv = self.likelihood.link_function(fx, params)
            ll = jnp.sum(rv.log_prob(y))

            log_prior_density = evaluate_priors(params, priors)
            constant = jnp.array(-1.0) if negative else jnp.array(1.0)
            return constant * (ll + log_prior_density)

        return mll


def construct_posterior(prior: Prior, likelihood: AbstractLikelihood) -> AbstractPosterior:
    if isinstance(likelihood, Conjugate):
        PosteriorGP = ConjugatePosterior

    elif isinstance(likelihood, NonConjugate):
        PosteriorGP = NonConjugatePosterior
    else:
        raise NotImplementedError(f"No posterior implemented for {likelihood.name} likelihood")
    return PosteriorGP(prior=prior, likelihood=likelihood)