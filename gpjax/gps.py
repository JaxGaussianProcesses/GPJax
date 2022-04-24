import typing as tp
from abc import abstractmethod, abstractproperty

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from chex import dataclass
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

from gpjax.config import get_defaults

from .kernels import Kernel, cross_covariance, gram, diagonal
from .likelihoods import (
    Gaussian,
    Likelihood,
    NonConjugateLikelihoods,
    NonConjugateLikelihoodType,
)
from .mean_functions import MeanFunction, Zero
from .parameters import copy_dict_structure, evaluate_priors, transform
from .types import Array, Dataset
from .utils import I, concat_dictionaries, chol_solve
from copy import deepcopy


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

    def __mul__(self, other: Likelihood):
        """The product of a prior and likelihood is proportional to the posterior distribution. By computing the product of a GP prior and a likelihood object, a posterior GP object will be returned.

        Args:
            other (Likelihood): The likelihood distribution of the observed dataset.

        Returns:
            Posterior: The relevant GP posterior for the given prior and likelihood. Special cases are accounted for where the model is conjugate.
        """
        return construct_posterior(prior=self, likelihood=other)

    def mean(self, params: dict) -> tp.Callable[[Array], Array]:
        """Compute the GP's prior mean function.

        Args:
            params (dict): The specific set of parameters for which the mean function should be defined for.

        Returns:
            tp.Callable[[Array], Array]: A mean function that accepts an input array for where the mean function should be evaluated at. The mean function's value at these points is then returned.
        """

        def mean_fn(test_points: Array) -> Array:
            mu = self.mean_function(test_points, params["mean_function"])
            return mu

        return mean_fn

    def variance(self, params: dict) -> tp.Callable[[Array], Array]:
        """Compute the GP's prior variance function.

        Args:
            params (dict): The specific set of parameters for which the variance function should be defined for.

        Returns:
            tp.Callable[[Array], Array]: A variance function that accepts an input array for where the variance function should be evaluated at. The variance function's value at these points is then returned as a covariance matrix.
        """

        def variance_fn(test_points: Array):
            Kff = gram(self.kernel, test_points, params["kernel"])
            jitter_matrix = I(test_points.shape[0]) * 1e-8
            covariance_matrix = Kff + jitter_matrix
            return covariance_matrix

        return variance_fn

    @property
    def params(self) -> dict:
        """Initialise the GP prior's parameter set"""
        return {
            "kernel": self.kernel.params,
            "mean_function": self.mean_function.params,
        }

    def random_variable(self, test_points: Array, params: dict) -> tfd.Distribution:
        """Using the GP's mean and covariance functions, we can also construct the multivariate normal random variable.

        Args:
            test_points (Array): The points at which we'd like to evaluate our mean and covariance function.
            params (dict): The parameterisation of the GP prior for which the random variable should be computed for.

        Returns:
            tfd.Distribution: A TensorFlow Probability Multivariate Normal distribution.
        """
        n = test_points.shape[0]
        mu = self.mean(params)(test_points)
        sigma = self.variance(params)(test_points)
        sigma += I(n) * 1e-8
        return tfd.MultivariateNormalTriL(mu.squeeze(), jnp.linalg.cholesky(sigma))


#######################
# GP Posteriors
#######################
@dataclass
class Posterior(GP):
    """The base GP posterior object conditioned on an observed dataset."""

    prior: Prior
    likelihood: Likelihood
    name: tp.Optional[str] = "GP Posterior"

    @abstractmethod
    def mean(self, training_data: Dataset, params: dict) -> tp.Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractmethod
    def variance(self, training_data: Dataset, params: dict) -> tp.Callable[[Dataset], Array]:
        raise NotImplementedError

    @property
    def params(self) -> dict:
        return concat_dictionaries(self.prior.params, {"likelihood": self.likelihood.params})


@dataclass
class ConjugatePosterior(Posterior):
    prior: Prior
    likelihood: Gaussian
    name: tp.Optional[str] = "ConjugatePosterior"

    def mean(self, training_data: Dataset, params: dict) -> tp.Callable[[Array], Array]:
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
            Kfx = cross_covariance(self.prior.kernel, test_inputs, X, params["kernel"])
            return prior_mean_at_test_inputs + jnp.dot(Kfx, weights)

        return mean_fn

    def variance(self, training_data: Dataset, params: dict) -> tp.Callable[[Array], Array]:
        X = training_data.X
        n_train = training_data.n
        obs_noise = params["likelihood"]["obs_noise"]
        n_train = training_data.n
        Kff = gram(self.prior.kernel, X, params["kernel"])
        Kff += I(n_train) * 1e-8
        L = cho_factor(Kff + I(n_train) * obs_noise, lower=True)

        def variance_fn(test_inputs: Array) -> Array:
            Kfx = cross_covariance(self.prior.kernel, test_inputs, X, params["kernel"])
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
            mu = self.prior.mean_function(x, params["mean_function"])
            gram_matrix = gram(self.prior.kernel, x, params["kernel"])
            gram_matrix += params["likelihood"]["obs_noise"] * I(x.shape[0])
            L = jnp.linalg.cholesky(gram_matrix)
            random_variable = tfd.MultivariateNormalTriL(mu, L)

            log_prior_density = evaluate_priors(params, priors)
            constant = jnp.array(-1.0) if negative else jnp.array(1.0)
            return constant * (random_variable.log_prob(y.squeeze()).mean() + log_prior_density)

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
        hyperparameters["latent"] = jnp.zeros(shape=(self.likelihood.num_datapoints, 1))
        return hyperparameters

    def mean(self, training_data: Dataset, params: dict) -> tp.Callable[[Dataset], Array]:
        X, y = training_data.X, training_data.y
        N = training_data.n
        Kff = gram(self.prior.kernel, X, params["kernel"])
        L = jnp.linalg.cholesky(Kff + I(N) * 1e-6)

        def meanf(test_inputs: Array) -> Array:
            Kfx = cross_covariance(self.prior.kernel, test_inputs, X, params["kernel"])
            Kxx = gram(self.prior.kernel, test_inputs, params["kernel"])
            A = solve_triangular(L, Kfx.T, lower=True)
            latent_var = Kxx - jnp.sum(jnp.square(A), -2)
            latent_mean = jnp.matmul(A.T, params["latent"])

            lvar = jnp.diag(latent_var)

            moment_fn = self.likelihood.predictive_moment_fn
            pred_rv = moment_fn(latent_mean.ravel(), lvar)
            return pred_rv.mean().reshape(-1, 1)

        return meanf

    def variance(self, training_data: Dataset, params: dict) -> tp.Callable[[Dataset], Array]:
        X, y = training_data.X, training_data.y
        N = training_data.n
        Kff = gram(self.prior.kernel, X, params["kernel"])
        L = jnp.linalg.cholesky(Kff + I(N) * 1e-6)

        def variancef(test_inputs: Array) -> Array:
            Kfx = cross_covariance(self.prior.kernel, test_inputs, X, params["kernel"])
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


#######################
# Sparse Approximations
#######################
@dataclass
class _ApproximateProcess:
    inducing_inputs: Array


@dataclass
class ApproximateGP(Posterior, _ApproximateProcess):
    def __post_init__(self):
        self.num_inducing = self.likelihood.num_datapoints

    @property
    def params(self) -> dict:
        hyperparams = concat_dictionaries(self.prior.params, {"likelihood": self.likelihood.params})
        hyperparams["inducing_inputs"] = deepcopy(self.inducing_inputs)
        return hyperparams

    def variational_dist(self, data: Dataset, params: dict):
        """Analytically compute the posterior mean and covariance

        Args:
            data (Dataset): _description_
            params (dict): _description_

        Returns:
            _type_: _description_
        """
        X, y, N = data.X, data.y, data.n
        prior_error = y - self.prior.mean_function(X, params["mean_function"])
        Z = params["inducing_inputs"]
        n_inducing = Z.shape[0]
        jitter = get_defaults()["jitter"]

        precision = 1.0 / params["likelihood"]["obs_noise"]
        Kzz = gram(self.prior.kernel, Z, params["kernel"]) + jitter * I(n_inducing)
        Kzx = cross_covariance(self.prior.kernel, Z, X, params["kernel"])

        sigma = Kzz + precision * Kzx @ jnp.transpose(Kzx)
        sqrt_sigma = jnp.linalg.cholesky(sigma)
        sqrt_sigma_kzz = solve_triangular(sqrt_sigma, Kzz)

        A = jnp.transpose(sqrt_sigma_kzz) @ sqrt_sigma_kzz
        m = (
            precision
            * jnp.transpose(sqrt_sigma_kzz)
            @ solve_triangular(sqrt_sigma, Kzx @ prior_error)
        )
        return m, A

    # def q(self, X_test, theta, X_m, mu_m, A_m, K_mm_inv):
    #     """
    #     Approximate posterior.

    #     Computes mean and covariance of latent
    #     function values at test inputs X_test.
    #     """

    #     K_ss = gram(self.prior.kernel, X_test, params["kernel"])
    #     K_sm = kernel(X_test, X_m, theta)
    #     K_ms = K_sm.T

    #     f_q = (K_sm @ K_mm_inv).dot(mu_m)
    #     f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms

    #     return f_q, f_q_cov

    def elbo(
        self,
        training: Dataset,
        transformations: tp.Dict,
        priors: dict = None,
        static_params: dict = None,
        negative: bool = False,
    ) -> tp.Callable[[Dataset], Array]:
        x, y = training.X, training.y
        n_obs = training.n
        jitter = get_defaults()["jitter"]
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)
        n_inducing = self.num_inducing
        constant = 0.5 * n_obs * jnp.log(2 * jnp.pi)

        def elbo_fn(params: dict) -> Array:
            params = transform(params=params, transform_map=transformations)
            Z = params["inducing_inputs"]
            beta = 1.0 / params["likelihood"]["obs_noise"]

            # Compute kernel matrices
            Kmm = gram(self.prior.kernel, Z, params["kernel"]) + I(n_inducing) * jitter
            Knm = cross_covariance(self.prior.kernel, Z, x, params["kernel"])
            # Kmn = jnp.transpose(Knm)
            Kff_diag = diagonal(self.prior.kernel, x, params["kernel"])

            L = jnp.linalg.cholesky(Kmm)
            A = solve_triangular(L, Knm, lower=True) * beta
            AAT = A @ jnp.transpose(A)
            B = I(self.num_inducing) + AAT
            LB = jnp.linalg.cholesky(B)

            c = solve_triangular(LB, A.dot(y), lower=True) * beta

            lb = -constant - jnp.sum(jnp.diag(LB))
            lb -= n_obs / 2 * jnp.log(params["likelihood"]["obs_noise"] ** 2)
            lb -= 0.5 * beta ** 2 * y.T.dot(y)
            lb += 0.5 * c.T.dot(c)
            lb -= 0.5 * beta ** 2 * jnp.sum(Kff_diag)
            lb += 0.5 * jnp.trace(AAT)
            return -lb.squeeze()

        return elbo_fn

    def mean(self, training_data: Dataset, params: dict) -> tp.Callable[[Array], Array]:
        X, y = training_data.X, training_data.y
        sigma = params["likelihood"]["obs_noise"]
        Z = params["inducing_inputs"]
        n_train = training_data.n

        def mean_fn(test_inputs: Array) -> Array:
            pass

        return mean_fn

    def variance(self, training_data: Dataset, params: dict) -> tp.Callable[[Array], Array]:
        X = training_data.X
        n_train = training_data.n
        obs_noise = params["likelihood"]["obs_noise"]
        n_train = training_data.n

        def variance_fn(test_inputs: Array) -> Array:
            pass

        return variance_fn


def construct_VFE_posterior(gp: ConjugatePosterior, Z: Array):
    num_inducing_points = Z.shape[0]
    assert isinstance(
        gp.likelihood, Gaussian
    ), "Variational free energy posterior is only defined for Gaussian likelihoods."
    prior_copy = deepcopy(gp.prior)
    likelihood = Gaussian(num_datapoints=num_inducing_points)
    return ApproximateGP(prior=prior_copy, likelihood=likelihood, inducing_inputs=Z)


def construct_posterior(prior: Prior, likelihood: Likelihood) -> Posterior:
    if isinstance(likelihood, Gaussian):
        PosteriorGP = ConjugatePosterior
    elif any([isinstance(likelihood, l) for l in NonConjugateLikelihoods]):
        PosteriorGP = NonConjugatePosterior
    else:
        raise NotImplementedError(f"No posterior implemented for {likelihood.name} likelihood")
    return PosteriorGP(prior=prior, likelihood=likelihood)
