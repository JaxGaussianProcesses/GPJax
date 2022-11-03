# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from abc import abstractmethod
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import distrax as dx
from chex import dataclass
from jaxtyping import Array, Float
import distrax as dx

from .config import get_defaults
from .covariance_operator import I
from .kernels import Kernel
from .likelihoods import AbstractLikelihood, Conjugate, Gaussian, NonConjugate
from .mean_functions import AbstractMeanFunction, Zero
from .parameters import copy_dict_structure, evaluate_priors
from .types import Dataset, PRNGKeyType
from .utils import concat_dictionaries

DEFAULT_JITTER = get_defaults()["jitter"]


@dataclass
class AbstractGP:
    """Abstract Gaussian process object."""

    def __call__(self, *args: Any, **kwargs: Any) -> dx.Distribution:
        """Evaluate the Gaussian process at the given points.

        Args:
            *args (Any): The arguments to pass to the GP's `predict` method.
            **kwargs (Any): The keyword arguments to pass to the GP's `predict` method.

        Returns:
            dx.Distribution: A multivariate normal random variable representation of the Gaussian process.
        """
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> dx.Distribution:
        """Compute the latent function's multivariate normal distribution.

        Args:
            *args (Any): Arguments to the predict method.
            **kwargs (Any): Keyword arguments to the predict method.

        Returns:
            dx.Distribution: A multivariate normal random variable representation of the Gaussian process.
        """
        raise NotImplementedError

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Initialise the GP's parameter set.

        Args:
            key (PRNGKeyType): The PRNG key.

        Returns:
            Dict: The initialised parameter set.
        """
        raise NotImplementedError


#######################
# GP Priors
#######################
@dataclass(repr=False)
class Prior(AbstractGP):
    """A Gaussian process prior object. The GP is parameterised by a mean and kernel function."""

    kernel: Kernel
    mean_function: Optional[AbstractMeanFunction] = Zero()
    name: Optional[str] = "GP prior"
    jitter: Optional[float] = DEFAULT_JITTER

    def __mul__(self, other: AbstractLikelihood):
        """The product of a prior and likelihood is proportional to the posterior distribution. By computing the product of a GP prior and a likelihood object, a posterior GP object will be returned.

        Args:
            other (Likelihood): The likelihood distribution of the observed dataset.

        Returns:
            Posterior: The relevant GP posterior for the given prior and likelihood. Special cases are accounted for where the model is conjugate.
        """
        return construct_posterior(prior=self, likelihood=other)

    def __rmul__(self, other: AbstractLikelihood):
        """Reimplement the multiplication operator to allow for order-invariant product of a likelihood and a prior i.e., likelihood * prior.

        Args:
            other (Likelihood): The likelihood distribution of the observed dataset.

        Returns:
            Posterior: The relevant GP posterior for the given prior and likelihood. Special cases are accounted for where the model is conjugate.
        """
        return self.__mul__(other)

    def predict(
        self, params: Dict
    ) -> Callable[[Float[Array, "N D"]], dx.MultivariateNormalTri]:
        """Compute the GP's prior mean and variance.

        Args:
            params (Dict): The specific set of parameters for which the mean function should be defined for.

        Returns:
            Callable[[Float[Array, "N D"]], dx.MultivariateNormalTri]: A mean function that accepts an input array for where the mean function should be evaluated at. The mean function's value at these points is then returned.
        """
        gram = self.kernel.gram

        def predict_fn(test_inputs: Float[Array, "N D"]) -> dx.MultivariateNormalTri:
            t = test_inputs
            n_test = t.shape[0]
            μt = self.mean_function(t, params["mean_function"])
            Ktt = gram(self.kernel, t, params["kernel"])
            Ktt += self.jitter * I(n_test)
            Lt = Ktt.triangular_lower()

            return dx.MultivariateNormalTri(jnp.atleast_1d(μt.squeeze()), Lt)

        return predict_fn

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Initialise the GP prior's parameter set.

        Args:
            key (PRNGKeyType): The PRNG key.

        Returns:
            Dict: The initialised parameter set.
        """
        return {
            "kernel": self.kernel._initialise_params(key),
            "mean_function": self.mean_function._initialise_params(key),
        }


#######################
# GP Posteriors
#######################
@dataclass
class AbstractPosterior(AbstractGP):
    """The base GP posterior object conditioned on an observed dataset."""

    prior: Prior
    likelihood: AbstractLikelihood
    name: Optional[str] = "GP posterior"
    jitter: Optional[float] = DEFAULT_JITTER

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> dx.Distribution:
        """Predict the GP's output given the input.

        Args:
            *args (Any): Arguments to the predict method.
            **kwargs (Any): Keyword arguments to the predict method.

        Returns:
            dx.Distribution: A multivariate normal random variable representation of the Gaussian process.
        """
        raise NotImplementedError

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Initialise the parameter set of a GP posterior.

        Args:
            key (PRNGKeyType): The PRNG key.

        Returns:
            Dict: The initialised parameter set.
        """
        return concat_dictionaries(
            self.prior._initialise_params(key),
            {"likelihood": self.likelihood._initialise_params(key)},
        )


@dataclass
class ConjugatePosterior(AbstractPosterior):
    """Gaussian process posterior object for models where the likelihood is Gaussian."""

    prior: Prior
    likelihood: Gaussian
    name: Optional[str] = "Conjugate posterior"
    jitter: Optional[float] = DEFAULT_JITTER

    def predict(
        self, train_data: Dataset, params: Dict
    ) -> Callable[[Float[Array, "N D"]], dx.MultivariateNormalTri]:
        """Conditional on a set of training data, compute the GP's posterior predictive distribution for a given set of parameters. The returned function can be evaluated at a set of test inputs to compute the corresponding predictive density.

        Args:
            train_data (Dataset): A `gpx.Dataset` object that contains the input and output data used for training dataset.
            params (Dict): A dictionary of parameters that should be used to compute the posterior.

        Returns:
            Callable[[Float[Array, "N D"]], dx.MultivariateNormalTri]: A function that accepts an input array and returns the predictive distribution as a `dx.MultivariateNormalTri`.
        """
        x, y, n = train_data.X, train_data.y, train_data.n
        gram, cross_covariance = (
            self.prior.kernel.gram,
            self.prior.kernel.cross_covariance,
        )

        # Observation noise σ²
        obs_noise = params["likelihood"]["obs_noise"]
        μx = self.prior.mean_function(x, params["mean_function"])

        # Precompute covariance matrices
        Kxx = gram(self.prior.kernel, x, params["kernel"])
        Kxx += I(n) * self.jitter

        # Σ = Kxx + Iσ²
        Sigma = Kxx + I(n) * obs_noise

        def predict(test_inputs: Float[Array, "N D"]) -> dx.Distribution:
            t = test_inputs
            n_test = t.shape[0]
            μt = self.prior.mean_function(t, params["mean_function"])
            Ktt = gram(self.prior.kernel, t, params["kernel"])
            Kxt = cross_covariance(self.prior.kernel, x, t, params["kernel"])

            # TODO: Investigate lower triangular solves for general covariance operators
            # this is more efficient than the full solve for dense matrices in the current implimentation.

            # Σ⁻¹ Kxt
            Sigma_inv_Kxt = Sigma.solve(Kxt)

            # μt  +  Ktx (Kxx + Iσ²)⁻¹ (y  -  μx)
            mean = μt + jnp.matmul(Sigma_inv_Kxt.T, y - μx)

            # Ktt  -  Ktx (Kxx + Iσ²)⁻¹ Kxt
            covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
            covariance += I(n_test) * self.jitter

            return dx.MultivariateNormalFullCovariance(
                jnp.atleast_1d(mean.squeeze()), covariance.to_dense()
            )

        return predict

    def marginal_log_likelihood(
        self,
        train_data: Dataset,
        priors: Dict = None,
        negative: bool = False,
    ) -> Callable[[Dict], Float[Array, "1"]]:
        """Compute the marginal log-likelihood function of the Gaussian process. The returned function can then be used for gradient based optimisation of the model's parameters or for model comparison. The implementation given here enables exact estimation of the Gaussian process' latent function values.

        Args:
            train_data (Dataset): The training dataset used to compute the marginal log-likelihood.
            priors (Dict, optional): _description_. Optional argument that contains the priors placed on the model's parameters. Defaults to None.
            negative (bool, optional): Whether or not the returned function should be negative. For optimisation, the negative is useful as minimisation of the negative marginal log-likelihood is equivalent to maximisation of the marginal log-likelihood. Defaults to False.

        Returns:
            Callable[[Dict], Float[Array, "1"]]: A functional representation of the marginal log-likelihood that can be evaluated at a given parameter set.
        """
        x, y, n = train_data.X, train_data.y, train_data.n
        gram = self.prior.kernel.gram

        def mll(
            params: Dict,
        ):
            # Observation noise σ²
            obs_noise = params["likelihood"]["obs_noise"]
            μx = self.prior.mean_function(x, params["mean_function"])
            Kxx = gram(self.prior.kernel, x, params["kernel"])
            Kxx += I(n) * self.jitter

            # TODO: This implementation does not take advantage of the covariance operator structure.
            # Future work concerns implementation of a custom Gaussian distribution / measure object that accepts a covariance operator.

            # Σ = (Kxx + Iσ²) = LLᵀ
            Sigma = Kxx + I(n) * obs_noise
            L = Sigma.triangular_lower()

            # p(y | x, θ), where θ are the model hyperparameters:

            marginal_likelihood = dx.MultivariateNormalTri(jnp.atleast_1d(μx.squeeze()), L)

            # log p(θ)
            log_prior_density = evaluate_priors(params, priors)

            constant = jnp.array(-1.0) if negative else jnp.array(1.0)
            return constant * (
                marginal_likelihood.log_prob(jnp.atleast_1d(y.squeeze())).squeeze()
                + log_prior_density
            )

        return mll


@dataclass
class NonConjugatePosterior(AbstractPosterior):
    """Generic Gaussian process posterior object for models where the likelihood is non-Gaussian."""

    prior: Prior
    likelihood: AbstractLikelihood
    name: Optional[str] = "Non-conjugate posterior"
    jitter: Optional[float] = DEFAULT_JITTER

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Initialise the parameter set of a non-conjugate GP posterior."""
        parameters = concat_dictionaries(
            self.prior._initialise_params(key),
            {"likelihood": self.likelihood._initialise_params(key)},
        )
        parameters["latent"] = jnp.zeros(shape=(self.likelihood.num_datapoints, 1))
        return parameters

    def predict(
        self, train_data: Dataset, params: Dict
    ) -> Callable[[Float[Array, "N D"]], dx.Distribution]:
        """Conditional on a set of training data, compute the GP's posterior predictive distribution for a given set of parameters. The returned function can be evaluated at a set of test inputs to compute the corresponding predictive density. Note, to gain predictions on the scale of the original data, the returned distribution will need to be transformed through the likelihood function's inverse link function.

                Args:
                    train_data (Dataset): A `gpx.Dataset` object that contains the input and output data used for training dataset.
                    params (Dict): A dictionary of parameters that should be used to compute the posterior.

                Returns:
        <<<<<<< HEAD
                    tp.Callable[[Array], dx.Distribution]: A function that accepts an input array and returns the predictive distribution as a `numpyro.distributions.MultivariateNormal`.
        """
        x, n = train_data.X, train_data.n
        gram, cross_covariance = (
            self.prior.kernel.gram,
            self.prior.kernel.cross_covariance,
        )

        Kxx = gram(self.prior.kernel, x, params["kernel"])
        Kxx += I(n) * self.jitter

        def predict_fn(test_inputs: Float[Array, "N D"]) -> dx.Distribution:
            t = test_inputs
            n_test = t.shape[0]
            Ktx = cross_covariance(self.prior.kernel, t, x, params["kernel"])
            Ktt = gram(self.prior.kernel, t, params["kernel"]) + I(n_test) * self.jitter
            μt = self.prior.mean_function(t, params["mean_function"])

            Lx = Kxx.triangular_lower()

            # Lx⁻¹ Kxt
            Lx_inv_Kxt = jsp.linalg.solve_triangular(Lx, Ktx, lower=True)

            # μt + Ktx Lx⁻¹ latent
            mean = μt + jnp.matmul(Lx_inv_Kxt.T, params["latent"])

            # Ktt - Ktx Kxx⁻¹ Kxt
            covariance = Ktt
            covariance += I(n_test) * self.jitter
            covariance = covariance.to_dense() - jnp.matmul(Lx_inv_Kxt.T, Lx_inv_Kxt)

            return dx.MultivariateNormalFullCovariance(jnp.atleast_1d(mean.squeeze()), covariance)

        return predict_fn

    def marginal_log_likelihood(
        self,
        train_data: Dataset,
        priors: Dict = None,
        negative: bool = False,
    ) -> Callable[[Dict], Float[Array, "1"]]:
        """Compute the marginal log-likelihood function of the Gaussian process. The returned function can then be used for gradient based optimisation of the model's parameters or for model comparison. The implementation given here is general and will work for any likelihood support by GPJax.

        Args:
            train_data (Dataset): The training dataset used to compute the marginal log-likelihood.
            priors (Dict, optional): _description_. Optional argument that contains the priors placed on the model's parameters. Defaults to None.
            negative (bool, optional): Whether or not the returned function should be negative. For optimisation, the negative is useful as minimisation of the negative marginal log-likelihood is equivalent to maximisation of the marginal log-likelihood. Defaults to False.

        Returns:
            Callable[[Dict], Float[Array, "1"]]: A functional representation of the marginal log-likelihood that can be evaluated at a given parameter set.
        """
        x, y, n = train_data.X, train_data.y, train_data.n
        gram = self.prior.kernel.gram
        if not priors:
            priors = copy_dict_structure(self._initialise_params(jr.PRNGKey(0)))
            priors["latent"] = dx.Normal(loc=0.0, scale=1.0)

        def mll(params: Dict):
            Kxx = gram(self.prior.kernel, x, params["kernel"])
            Kxx += I(n) * self.jitter
            Lx = Kxx.triangular_lower()
            μx = self.prior.mean_function(x, params["mean_function"])

            # f(x) = μx  +  Lx latent
            fx = μx + jnp.matmul(Lx, params["latent"])

            # p(y | f(x), θ), where θ are the model hyperparameters:
            likelihood = self.likelihood.link_function(fx, params)

            # log p(θ)
            log_prior_density = evaluate_priors(params, priors)

            constant = jnp.array(-1.0) if negative else jnp.array(1.0)
            return constant * (likelihood.log_prob(y).sum() + log_prior_density)

        return mll


def construct_posterior(prior: Prior, likelihood: AbstractLikelihood) -> AbstractPosterior:
    if isinstance(likelihood, Conjugate):
        PosteriorGP = ConjugatePosterior

    elif isinstance(likelihood, NonConjugate):
        PosteriorGP = NonConjugatePosterior
    else:
        raise NotImplementedError(f"No posterior implemented for {likelihood.name} likelihood")
    return PosteriorGP(prior=prior, likelihood=likelihood)


def euclidean_distance(
    x: Float[Array, "N D"], y: Float[Array, "N D"]
) -> Float[Array, "N"]:
    """Compute the Euclidean distance between two arrays of points.

    Args:
        x (Float[Array, "N D"]): An array of points.
        y (Float[Array, "N D"]): An array of points.

    Returns:
        Float[Array, "N"]: An array of distances.
    """
    return jnp.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


__all__ = [
    AbstractGP,
    Prior,
    AbstractPosterior,
    ConjugatePosterior,
    NonConjugatePosterior,
    construct_posterior,
]
