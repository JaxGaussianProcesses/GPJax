from abc import abstractmethod
from typing import Optional, Callable
from typing_extensions import ParamSpec

from chex import dataclass
from multipledispatch import dispatch
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
import jax.numpy as jnp
import jax.random as jr

from .config import get_defaults
from .types import Dataset, Array
from .kernels import Kernel, gram, cross_covariance
from .kernels.spectral import SpectralKernel
from .likelihoods import (
    Gaussian,
    Likelihood,
    NonConjugateLikelihoods,
    NonConjugateLikelihoodType,
)
from .mean_functions import MeanFunction, Zero
from .utils import I, concat_dictionaries


@dataclass
class GP:
    kernel: Kernel
    mean_function: Optional[MeanFunction] = Zero()
    name: Optional[str] = "Gaussian process"

    @abstractmethod
    def mean(self) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @property
    def params(self) -> dict:
        return {
            "kernel": self.kernel.params,
            "mean_function": self.mean_function.params,
        }


#######################
# GP Priors
#######################
@dataclass(repr=False)
class Prior(GP):
    kernel: Kernel
    mean_function: Optional[MeanFunction] = Zero()
    name: Optional[str] = "Prior"

    def __mul__(self, other: Gaussian):
        return construct_posterior(prior=self, likelihood=Likelihood)

    def mean(self) -> Callable[[Dataset], Array]:
        def mean_fn(data: Dataset, params: dict):
            X = data.X
            mu = self.mean_function(X)
            return mu

        return mean_fn

    def variance(self) -> Callable[[Dataset], Array]:
        def variance_fn(data: Dataset, params: dict):
            X = data.X
            n_data = data.n
            Kff = gram(self.kernel, X, params["kernel"])
            jitter_matrix = I(n_data) * 1e-8
            covariance_matrix = Kff + jitter_matrix
            return covariance_matrix

        return variance_fn


#######################
# GP Posteriors
#######################
@dataclass
class Posterior:
    prior: Prior
    likelihood: Likelihood
    name: Optional[str] = "GP Posterior"

    @abstractmethod
    def mean(self, training_data: Dataset, params: dict) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractmethod
    def variance(
        self, training_data: Dataset, params: dict
    ) -> Callable[[Dataset], Array]:
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
    name: Optional[str] = "ConjugatePosterior"

    # def __repr__(self):
    #     meanf_string = self.prior.mean_function.__repr__()
    #     kernel_string = self.prior.kernel.__repr__()
    #     likelihood_string = self.likelihood.__repr__()
    #     return f"Conjugate Posterior\n{'-'*80}\n- {meanf_string}\n- {kernel_string}\n- {likelihood_string}"

    @abstractmethod
    def mean(self, training_data: Dataset, params: dict) -> Callable[[Dataset], Array]:
        X, y = training_data.X, training_data.y
        sigma = params["likelihood"]["obs_noise"]
        n_train = training_data.n
        # Precompute covariance matrices
        Kff = gram(self.prior.kernel, X, params["kernel"])
        prior_mean = self.prior.mean_function(X, params["mean_function"])
        L = cho_factor(Kff + I(n_train) * sigma, lower=True)

        prior_distance = y - prior_mean
        weights = cho_solve(L, prior_distance)

        def meanf(test_inputs: Array) -> Array:
            prior_mean_at_test_inputs = self.prior.mean_function(test_inputs)
            Kfx = cross_covariance(self.prior.kernel, X, test_inputs, params["kernel"])
            return prior_mean_at_test_inputs + jnp.dot(Kfx, weights)

        return meanf

    def variance(
        self, training_data: Dataset, params: dict
    ) -> Callable[[Dataset], Array]:
        X = training_data.X
        n_train = training_data.n
        obs_noise = params["likelihood"]["obs_noise"]
        n_train = training_data.n
        Kff = gram(self.prior.kernel, X, params["kernel"])
        Kff += I(n_train) * 1e-8
        L = cho_factor(Kff + I(n_train) * obs_noise, lower=True)

        def variance_fn(test_inputs: Dataset) -> Array:
            x_test = test_inputs.X
            Kfx = cross_covariance(self.prior.kernel, X, x_test, params["kernel"])
            Kxx = gram(self.prior.kernel, x_test, params["kernel"])
            latents = cho_solve(L, Kfx.T)
            return Kxx - jnp.dot(Kfx, latents)

        return variance_fn


@dataclass
class NonConjugatePosterior:
    prior: Prior
    likelihood: NonConjugateLikelihoodType
    name: Optional[str] = "ConjugatePosterior"

    def __repr__(self):
        meanf_string = self.prior.mean_function.__repr__()
        kernel_string = self.prior.kernel.__repr__()
        likelihood_string = self.likelihood.__repr__()
        return f"Conjugate Posterior\n{'-'*80}\n- {meanf_string}\n- {kernel_string}\n- {likelihood_string}"

    @property
    def params(self) -> dict:
        hyperparams = concat_dictionaries(
            self.prior.params, {"likelihood": self.likelihood.params}
        )
        hyperparams["latent"] = jnp.zeros(shape=(self.likelihood.num_datapoints, 1))
        return hyperparams


@dataclass
class SpectralPosterior:
    prior: Prior
    likelihood: Gaussian
    name: Optional[str] = "SpectralPosterior"

    # def __repr__(self):
    #     meanf_string = self.prior.mean_function.__repr__()
    #     kernel_string = self.prior.kernel.__repr__()
    #     likelihood_string = self.likelihood.__repr__()
    #     return f"Sparse Spectral Posterior\n{'-'*80}\n- {meanf_string}\n- {kernel_string}\n- {likelihood_string}"

    def mean(self, training_data: Dataset, params: dict) -> Callable[[Dataset], Array]:
        X, y = training_data.X, training_data.y
        N = training_data.n
        Kff = gram(self.prior.kernel, X, params["kernel"])
        L = jnp.linalg.cholesky(Kff + I(N) * 1e-6)

        def mean_fn(test_inputs: Dataset) -> Array:
            x_test = test_inputs.X
            Kfx = cross_covariance(self.prior.kernel, X, x_test, params["kernel"])
            Kxx = gram(self.prior.kernel, x_test, params["kernel"])
            A = solve_triangular(L, Kfx.T, lower=True)
            latent_var = Kxx - jnp.sum(jnp.square(A), -2)
            latent_mean = jnp.matmul(A.T, self.latent_vals)

            lvar = jnp.diag(latent_var)

            pred_rv = self.likelihood.predictive_moment_fn(latent_mean.ravel(), lvar)
            return pred_rv.mean()

        return mean_fn

    def variance(
        self, training_data: Dataset, params: dict
    ) -> Callable[[Dataset], Array]:
        X, y = training_data.X, training_data.y
        N = training_data.n
        Kff = gram(self.prior.kernel, X, params["kernel"])
        L = jnp.linalg.cholesky(Kff + I(N) * 1e-8)

        def variance_fn(test_inputs: Dataset) -> Array:
            x_test = test_inputs.X
            Kfx = cross_covariance(self.prior.kernel, X, x_test, params["kernel"])
            Kxx = gram(self.prior.kernel, x_test, params["kernel"])
            A = solve_triangular(L, Kfx.T, lower=True)
            latent_var = Kxx - jnp.sum(jnp.square(A), -2)
            latent_mean = jnp.matmul(A.T, self.latent_vals)

            lvar = jnp.diag(latent_var)

            pred_rv = self.likelihood.predictive_moment_fn(latent_mean.ravel(), lvar)
            return pred_rv.variance()

        return variance_fn


def construct_posterior(prior: Prior, likelihood: Likelihood) -> Posterior:
    if isinstance(likelihood, Gaussian):
        return ConjugatePosterior(prior, likelihood)
    elif likelihood in NonConjugateLikelihoods:
        return NonConjugatePosterior(prior, likelihood)
    else:
        raise NotImplementedError(
            f"No posterior implemented for {likelihood.name} likelihood"
        )
