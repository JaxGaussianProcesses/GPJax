import typing as tp
from abc import abstractmethod, abstractproperty

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from chex import dataclass
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

from gpjax.config import get_defaults

from .kernels import Kernel, cross_covariance, gram
from .likelihoods import (
    Gaussian,
    Likelihood,
    NonConjugateLikelihoods,
    NonConjugateLikelihoodType,
)
from .mean_functions import MeanFunction, Zero
from .parameters import evaluate_priors, copy_dict_structure, transform
from .types import Array, Dataset
from .utils import I, concat_dictionaries


@dataclass
class GP:
    @abstractmethod
    def mean(self) -> tp.Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> tp.Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractproperty
    def params(self) -> tp.Dict:
        raise NotImplementedError


#######################
# GP Priors
#######################
@dataclass(repr=False)
class Prior(GP):
    kernel: Kernel
    mean_function: tp.Optional[MeanFunction] = Zero()
    name: tp.Optional[str] = "Prior"

    def __mul__(self, other: Gaussian):
        return construct_posterior(prior=self, likelihood=other)

    def mean(self, params: dict) -> tp.Callable[[Array], Array]:
        def mean_fn(test_points: Array):
            mu = self.mean_function(test_points, params["mean_function"])
            return mu

        return mean_fn

    def variance(self, params: dict) -> tp.Callable[[Array], Array]:
        def variance_fn(test_points: Array):
            Kff = gram(self.kernel, test_points, params["kernel"])
            jitter_matrix = I(test_points.shape[0]) * 1e-8
            covariance_matrix = Kff + jitter_matrix
            return covariance_matrix

        return variance_fn

    @property
    def params(self) -> dict:
        return {
            "kernel": self.kernel.params,
            "mean_function": self.mean_function.params,
        }

    def random_variable(
        self, test_points: Array, params: dict
    ) -> tfd.Distribution:
        n = test_points.shape[0]
        mu = self.mean(params)(test_points)
        sigma = self.variance(params)(test_points)
        sigma += I(n) * 1e-8
        return tfd.MultivariateNormalTriL(
            mu.squeeze(), jnp.linalg.cholesky(sigma)
        )


#######################
# GP Posteriors
#######################
@dataclass
class Posterior(GP):
    prior: Prior
    likelihood: Likelihood
    name: tp.Optional[str] = "GP Posterior"

    @abstractmethod
    def mean(
        self, training_data: Dataset, params: dict
    ) -> tp.Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractmethod
    def variance(
        self, training_data: Dataset, params: dict
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

    def mean(
        self, training_data: Dataset, params: dict
    ) -> tp.Callable[[Array], Array]:
        X, y = training_data.X, training_data.y
        sigma = params["likelihood"]["obs_noise"]
        n_train = training_data.n
        # Precompute covariance matrices
        Kff = gram(self.prior.kernel, X, params["kernel"])
        prior_mean = self.prior.mean_function(X, params["mean_function"])
        L = cho_factor(Kff + I(n_train) * sigma, lower=True)

        prior_distance = y - prior_mean
        weights = cho_solve(L, prior_distance)

        def mean_fn(test_inputs: Array) -> Array:
            prior_mean_at_test_inputs = self.prior.mean_function(
                test_inputs, params["mean_function"]
            )
            Kfx = cross_covariance(
                self.prior.kernel, X, test_inputs, params["kernel"]
            )
            return prior_mean_at_test_inputs + jnp.dot(Kfx, weights)

        return mean_fn

    def variance(
        self, training_data: Dataset, params: dict
    ) -> tp.Callable[[Array], Array]:
        X = training_data.X
        n_train = training_data.n
        obs_noise = params["likelihood"]["obs_noise"]
        n_train = training_data.n
        Kff = gram(self.prior.kernel, X, params["kernel"])
        Kff += I(n_train) * 1e-8
        L = cho_factor(Kff + I(n_train) * obs_noise, lower=True)

        def variance_fn(test_inputs: Array) -> Array:
            Kfx = cross_covariance(
                self.prior.kernel, X, test_inputs, params["kernel"]
            )
            Kxx = gram(self.prior.kernel, test_inputs, params["kernel"])
            latent_values = cho_solve(L, Kfx.T)
            return Kxx - jnp.dot(Kfx, latent_values)

        return variance_fn

    def marginal_log_likelihood(
        self,
        training: Dataset,
        transformations: tp.Dict,
        priors: dict = None,
        static_params: dict = None,
        negative: bool = False,
    ) -> tp.Callable[[Dataset], Array]:
        x, y = training.X, training.y

        def mll(
            params: dict,
        ):
            params = transform(params=params, transform_map=transformations)
            if static_params:
                params = concat_dictionaries(params, transform(static_params))
            mu = self.prior.mean_function(x, params)
            gram_matrix = gram(self.prior.kernel, x, params["kernel"])
            gram_matrix += params["likelihood"]["obs_noise"] * I(x.shape[0])
            L = jnp.linalg.cholesky(gram_matrix)
            random_variable = tfd.MultivariateNormalTriL(mu, L)

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

    def __repr__(self):
        mean_fn_string = self.prior.mean_function.__repr__()
        kernel_string = self.prior.kernel.__repr__()
        likelihood_string = self.likelihood.__repr__()
        return (
            f"Conjugate Posterior\n{'-'*80}\n- {mean_fn_string}\n-"
            f" {kernel_string}\n- {likelihood_string}"
        )

    @property
    def params(self) -> dict:
        hyperparameters = concat_dictionaries(
            self.prior.params, {"likelihood": self.likelihood.params}
        )
        hyperparameters["latent"] = jnp.zeros(
            shape=(self.likelihood.num_datapoints, 1)
        )
        return hyperparameters

    def mean(
        self, training_data: Dataset, params: dict
    ) -> tp.Callable[[Dataset], Array]:
        X, y = training_data.X, training_data.y
        N = training_data.n
        Kff = gram(self.prior.kernel, X, params["kernel"])
        L = jnp.linalg.cholesky(Kff + I(N) * 1e-6)

        def meanf(test_inputs: Array) -> Array:
            Kfx = cross_covariance(
                self.prior.kernel, X, test_inputs, params["kernel"]
            )
            Kxx = gram(self.prior.kernel, test_inputs, params["kernel"])
            A = solve_triangular(L, Kfx.T, lower=True)
            latent_var = Kxx - jnp.sum(jnp.square(A), -2)
            latent_mean = jnp.matmul(A.T, params["latent"])

            lvar = jnp.diag(latent_var)

            moment_fn = self.likelihood.predictive_moment_fn
            pred_rv = moment_fn(latent_mean.ravel(), lvar)
            return pred_rv.mean().reshape(-1, 1)

        return meanf

    def variance(
        self, training_data: Dataset, params: dict
    ) -> tp.Callable[[Dataset], Array]:
        X, y = training_data.X, training_data.y
        N = training_data.n
        Kff = gram(self.prior.kernel, X, params["kernel"])
        L = jnp.linalg.cholesky(Kff + I(N) * 1e-6)

        def variancef(test_inputs: Array) -> Array:
            Kfx = cross_covariance(
                self.prior.kernel, X, test_inputs, params["kernel"]
            )
            Kxx = gram(self.prior.kernel, test_inputs, params["kernel"])
            A = solve_triangular(L, Kfx.T, lower=True)
            latent_var = Kxx - jnp.sum(jnp.square(A), -2)
            latent_mean = jnp.matmul(A.T, params["latent"])

            lvar = jnp.diag(latent_var)

            moment_fn = self.likelihood.predictive_moment_fn
            pred_rv = moment_fn(latent_mean.ravel(), lvar)
            return jnp.diag(pred_rv.variance())

        return variancef

    def marginal_log_likelihood(
        self,
        training: Dataset,
        transformations: tp.Dict,
        priors: dict = None,
        static_params: dict = None,
        negative: bool = False,
    ) -> tp.Callable[[Dataset], Array]:
        x, y = training.X, training.y
        n = training.n
        jitter = get_defaults()["jitter"]
        if not priors:
            priors = copy_dict_structure(self.params)
            priors["latent"] = tfd.Normal(loc=0.0, scale=1.0)

        def mll(params: dict):
            params = transform(params=params, transform_map=transformations)
            if static_params:
                params = concat_dictionaries(params, transform(static_params))
            gram_matrix = gram(self.prior.kernel, x, params["kernel"])
            gram_matrix += I(n) * jitter
            L = jnp.linalg.cholesky(gram_matrix)
            F = jnp.matmul(L, params["latent"])
            rv = self.likelihood.link_function(F)
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
