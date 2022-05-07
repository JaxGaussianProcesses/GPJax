import typing as tp
from abc import abstractmethod, abstractproperty

import distrax as dx
import jax.numpy as jnp
from chex import dataclass
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

from .config import get_defaults
from .kernels import Kernel, cross_covariance, gram
from .likelihoods import (
    Gaussian,
    Likelihood,
    NonConjugateLikelihoods,
    NonConjugateLikelihoodType,
)
from .mean_functions import MeanFunction, Zero
from .parameters import copy_dict_structure, evaluate_priors, transform
from .types import Array, Dataset
from .utils import I, concat_dictionaries

DEFAULT_JITTER = get_defaults()["jitter"]


@dataclass
class GP:
    """Abstract Gaussian process object."""

    @abstractmethod
    def mean(self) -> tp.Callable[[Dataset], Array]:
        """Compute the GP's mean function."""
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> tp.Callable[[Dataset], Array]:
        """Compute the GP's variance function."""
        raise NotImplementedError

    @abstractproperty
    def params(self) -> tp.Dict:
        """Initialise the GP's parameter set"""
        raise NotImplementedError


#######################
# GP Priors
#######################
@dataclass(repr=False)
class Prior(GP):
    """A Gaussian process prior object. The GP is parameterised by a mean and kernel function."""

    kernel: Kernel
    mean_function: tp.Optional[MeanFunction] = Zero()
    name: tp.Optional[str] = "Prior"
    jitter: tp.Optional[float] = DEFAULT_JITTER

    def __mul__(self, other: Gaussian):
        """The product of a prior and likelihood is proportional to the posterior distribution. By computing the product of a GP prior and a likelihood object, a posterior GP object will be returned.
        Args:
            other (Likelihood): The likelihood distribution of the observed dataset.
        Returns:
            Posterior: The relevant GP posterior for the given prior and likelihood. Special cases are accounted for where the model is conjugate.
        """
        return construct_posterior(prior=self, likelihood=other)

    def __rmul__(self, other: Gaussian):
        """Reimplement the multiplication operator to allow for order-invariant product of a likelihood and a prior i.e., ."""
        return self.__mul__(other)

    def mean(self, params: dict) -> tp.Callable[[Array], Array]:
        """Compute the GP's prior mean function.
        Args:
            params (dict): The specific set of parameters for which the mean function should be defined for.
        Returns:
            tp.Callable[[Array], Array]: A mean function that accepts an input array for where the mean function should be evaluated at. The mean function's value at these points is then returned.
        """

        def mean_fn(test_inputs: Array):
            t = test_inputs
            mt = self.mean_function(t, params["mean_function"])
            return mt

        return mean_fn

    def variance(self, params: dict) -> tp.Callable[[Array], Array]:
        """Compute the GP's prior variance function.
        Args:
            params (dict): The specific set of parameters for which the variance function should be defined for.
        Returns:
            tp.Callable[[Array], Array]: A variance function that accepts an input array for where the variance function should be evaluated at. The variance function's value at these points is then returned as a covariance matrix.
        """

        def variance_fn(test_inputs: Array):
            t = test_inputs
            Ktt = gram(self.kernel, t, params["kernel"])
            return Ktt

        return variance_fn

    @property
    def params(self) -> dict:
        """Initialise the GP prior's parameter set"""
        return {
            "kernel": self.kernel.params,
            "mean_function": self.mean_function.params,
        }

    def random_variable(self, test_inputs: Array, params: dict) -> dx.Distribution:
        """Using the GP's mean and covariance functions, we can also construct the multivariate normal random variable.
        Args:
            test_points (Array): The points at which we'd like to evaluate our mean and covariance function.
            params (dict): The parameterisation of the GP prior for which the random variable should be computed for.
        Returns:
            dx.Distribution: A Distrax Multivariate Normal distribution.
        """
        t = test_inputs
        nt = t.shape[0]
        mt = self.mean(params)(t)
        Ktt = self.variance(params)(t)
        Ktt += I(nt) * self.jitter
        Lt = jnp.linalg.cholesky(Ktt)
        return dx.MultivariateNormalTri(jnp.atleast_1d(mt.squeeze()), Lt)


#######################
# GP Posteriors
#######################
@dataclass
class Posterior(GP):
    """The base GP posterior object conditioned on an observed dataset."""

    prior: Prior
    likelihood: Likelihood
    name: tp.Optional[str] = "GP Posterior"
    jitter: tp.Optional[float] = DEFAULT_JITTER

    @abstractmethod
    def mean(self, train_data: Dataset, params: dict) -> tp.Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractmethod
    def variance(
        self, train_data: Dataset, params: dict
    ) -> tp.Callable[[Dataset], Array]:
        raise NotImplementedError

    @property
    def params(self) -> dict:
        return concat_dictionaries(
            self.prior.params, {"likelihood": self.likelihood.params}
        )


@dataclass
class ConjugatePosterior(Posterior):
    prior: Prior
    likelihood: Gaussian
    name: tp.Optional[str] = "ConjugatePosterior"
    jitter: tp.Optional[float] = DEFAULT_JITTER

    def mean(self, train_data: Dataset, params: dict) -> tp.Callable[[Array], Array]:
        x, y, n_data = train_data.X, train_data.y, train_data.n
        obs_noise = params["likelihood"]["obs_noise"]
        mx = self.prior.mean_function(x, params["mean_function"])

        # Precompute covariance matrices
        Kxx = gram(self.prior.kernel, x, params["kernel"])
        Kxx += I(n_data) * self.jitter
        Lx = cho_factor(Kxx + I(n_data) * obs_noise, lower=True)

        weights = cho_solve(Lx, y - mx)

        def mean_fn(test_inputs: Array) -> Array:
            t = test_inputs
            mt = self.prior.mean_function(t, params["mean_function"])
            Ktx = cross_covariance(self.prior.kernel, t, x, params["kernel"])
            return mt + jnp.dot(Ktx, weights)

        return mean_fn

    def variance(
        self, train_data: Dataset, params: dict
    ) -> tp.Callable[[Array], Array]:
        x, n_data = train_data.X, train_data.n
        obs_noise = params["likelihood"]["obs_noise"]
        Kxx = gram(self.prior.kernel, x, params["kernel"])
        Kxx += I(n_data) * self.jitter
        Lx = cho_factor(Kxx + I(n_data) * obs_noise, lower=True)

        def variance_fn(test_inputs: Array) -> Array:
            t = test_inputs
            Ktx = cross_covariance(self.prior.kernel, t, x, params["kernel"])
            Ktt = gram(self.prior.kernel, t, params["kernel"])
            latent_values = cho_solve(Lx, Ktx.T)
            return Ktt - jnp.dot(Ktx, latent_values)

        return variance_fn

    def marginal_log_likelihood(
        self,
        train_data: Dataset,
        transformations: tp.Dict,
        priors: dict = None,
        negative: bool = False,
    ) -> tp.Callable[[Dataset], Array]:
        x, y, n_data = train_data.X, train_data.y, train_data.n

        def mll(
            params: dict,
        ):
            params = transform(params=params, transform_map=transformations)

            obs_noise = params["likelihood"]["obs_noise"]
            mu = self.prior.mean_function(x, params)
            Kxx = gram(self.prior.kernel, x, params["kernel"])
            Kxx += I(n_data) * self.jitter
            Lx = jnp.linalg.cholesky(Kxx + I(n_data) * obs_noise)

            random_variable = dx.MultivariateNormalTri(mu.squeeze(), Lx)

            log_prior_density = evaluate_priors(params, priors)
            constant = jnp.array(-1.0) if negative else jnp.array(1.0)
            return constant * (
                random_variable.log_prob(y.squeeze()).mean() + log_prior_density
            )

        return mll


@dataclass
class NonConjugatePosterior(Posterior):
    prior: Prior
    likelihood: NonConjugateLikelihoodType
    name: tp.Optional[str] = "Non-Conjugate Posterior"
    jitter: tp.Optional[float] = DEFAULT_JITTER

    def __repr__(self):
        mean_fn_string = self.prior.mean_function.__repr__()
        kernel_string = self.prior.kernel.__repr__()
        likelihood_string = self.likelihood.__repr__()
        return (
            f"Non-Conjugate Posterior\n{'-'*80}\n- {mean_fn_string}\n-"
            f" {kernel_string}\n- {likelihood_string}"
        )

    @property
    def params(self) -> dict:
        hyperparameters = concat_dictionaries(
            self.prior.params, {"likelihood": self.likelihood.params}
        )
        hyperparameters["latent"] = jnp.zeros(shape=(self.likelihood.num_datapoints, 1))
        return hyperparameters

    def mean(self, train_data: Dataset, params: dict) -> tp.Callable[[Dataset], Array]:
        x, n_data = train_data.X, train_data.n
        Kxx = gram(self.prior.kernel, x, params["kernel"])
        Kxx += I(n_data) * self.jitter
        Lx = jnp.linalg.cholesky(Kxx)

        def mean_fn(test_inputs: Array) -> Array:
            t = test_inputs
            Ktx = cross_covariance(self.prior.kernel, t, x, params["kernel"])
            Ktt = gram(self.prior.kernel, t, params["kernel"])
            A = solve_triangular(Lx, Ktx.T, lower=True)
            latent_var = Ktt - jnp.sum(jnp.square(A), -2)
            latent_mean = jnp.matmul(A.T, params["latent"])

            lvar = jnp.diag(latent_var)

            moment_fn = self.likelihood.predictive_moment_fn
            pred_rv = moment_fn(latent_mean.ravel(), lvar)
            return pred_rv.mean().reshape(-1, 1)

        return mean_fn

    def variance(
        self, train_data: Dataset, params: dict
    ) -> tp.Callable[[Dataset], Array]:
        x, n_data = train_data.X, train_data.n
        Kxx = gram(self.prior.kernel, x, params["kernel"])
        Lx = jnp.linalg.cholesky(Kxx + I(n_data) * self.jitter)

        def variance_fn(test_inputs: Array) -> Array:
            t = test_inputs
            Ktx = cross_covariance(self.prior.kernel, t, x, params["kernel"])
            Ktt = gram(self.prior.kernel, t, params["kernel"])
            A = solve_triangular(Lx, Ktx.T, lower=True)
            latent_var = Ktt - jnp.sum(jnp.square(A), -2)
            latent_mean = jnp.matmul(A.T, params["latent"])

            lvar = jnp.diag(latent_var)

            moment_fn = self.likelihood.predictive_moment_fn
            pred_rv = moment_fn(latent_mean.ravel(), lvar)
            return jnp.diag(pred_rv.variance())

        return variance_fn

    def marginal_log_likelihood(
        self,
        train_data: Dataset,
        transformations: tp.Dict,
        priors: dict = None,
        negative: bool = False,
    ) -> tp.Callable[[Dataset], Array]:
        x, y, n_data = train_data.X, train_data.y, train_data.n

        if not priors:
            priors = copy_dict_structure(self.params)
            priors["latent"] = dx.Normal(loc=0.0, scale=1.0)

        def mll(params: dict):
            params = transform(params=params, transform_map=transformations)
            Kxx = gram(self.prior.kernel, x, params["kernel"])
            Kxx += I(n_data) * self.jitter
            Lx = jnp.linalg.cholesky(Kxx)
            Fx = jnp.matmul(Lx, params["latent"])
            rv = self.likelihood.link_function(Fx, params)
            ll = jnp.sum(rv.log_prob(y))

            log_prior_density = evaluate_priors(params, priors)
            constant = jnp.array(-1.0) if negative else jnp.array(1.0)
            return constant * (ll + log_prior_density)

        return mll


def construct_posterior(prior: Prior, likelihood: Likelihood) -> Posterior:
    if isinstance(likelihood, Gaussian):
        PosteriorGP = ConjugatePosterior
    elif any([isinstance(likelihood, l) for l in NonConjugateLikelihoods]):
        PosteriorGP = NonConjugatePosterior
    else:
        raise NotImplementedError(
            f"No posterior implemented for {likelihood.name} likelihood"
        )
    return PosteriorGP(prior=prior, likelihood=likelihood)
