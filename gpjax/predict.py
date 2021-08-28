from typing import Callable

import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from multipledispatch import dispatch

from .gps import ConjugatePosterior, NonConjugatePosterior, Prior
from .kernels import cross_covariance, gram
from .likelihoods import predictive_moments
from .types import Array, Dataset
from .utils import I


@dispatch(ConjugatePosterior, dict, Dataset)
def mean(gp: ConjugatePosterior, param: dict, training: Dataset) -> Callable:
    X, y = training.X, training.y
    sigma = param["obs_noise"]
    n_train = training.n
    # Precompute covariance matrices
    Kff = gram(gp.prior.kernel, X, param)
    prior_mean = gp.prior.mean_function(X)
    L = cho_factor(Kff + I(n_train) * sigma, lower=True)

    prior_distance = y - prior_mean
    weights = cho_solve(L, prior_distance)

    def meanf(test_inputs: Array) -> Array:
        prior_mean_at_test_inputs = gp.prior.mean_function(test_inputs)
        Kfx = cross_covariance(gp.prior.kernel, X, test_inputs, param)
        return prior_mean_at_test_inputs + jnp.dot(Kfx, weights)

    return meanf


@dispatch(ConjugatePosterior, dict, Dataset)
def variance(
    gp: ConjugatePosterior,
    param: dict,
    training: Dataset,
) -> Callable:
    X, y = training.X, training.y
    sigma = param["obs_noise"]
    n_train = training.n
    Kff = gram(gp.prior.kernel, X, param)
    L = cho_factor(Kff + I(n_train) * sigma, lower=True)

    def varf(test_inputs: Array) -> Array:
        Kfx = cross_covariance(gp.prior.kernel, X, test_inputs, param)
        Kxx = gram(gp.prior.kernel, test_inputs, param)
        latents = cho_solve(L, Kfx.T)
        return Kxx - jnp.dot(Kfx, latents)

    return varf


@dispatch(NonConjugatePosterior, dict, Dataset)
def mean(gp: NonConjugatePosterior, param: dict, training: Dataset) -> Array:
    ell, alpha, nu = param["lengthscale"], param["variance"], param["latent"]
    X, y = training.X, training.y
    N = training.n
    Kff = gram(gp.prior.kernel, X, param)
    L = jnp.linalg.cholesky(Kff + I(N) * 1e-6)

    def meanf(test_inputs: Array) -> Array:
        Kfx = cross_covariance(gp.prior.kernel, X, test_inputs, param)
        Kxx = gram(gp.prior.kernel, test_inputs, param)
        A = solve_triangular(L, Kfx.T, lower=True)
        latent_var = Kxx - jnp.sum(jnp.square(A), -2)
        latent_mean = jnp.matmul(A.T, nu)

        lvar = jnp.diag(latent_var)

        moment_fn = predictive_moments(gp.likelihood)
        pred_rv = moment_fn(latent_mean.ravel(), lvar)
        return pred_rv.mean()

    return meanf


@dispatch(NonConjugatePosterior, dict, Dataset)
def variance(gp: NonConjugatePosterior, param: dict, training: Dataset):
    X, y = training.X, training.y
    N = training.n
    ell, alpha, nu = param["lengthscale"], param["variance"], param["latent"]
    Kff = gram(gp.prior.kernel, X, param)
    L = jnp.linalg.cholesky(Kff + I(N) * 1e-6)

    def variancef(test_inputs: Array) -> Array:
        Kfx = cross_covariance(gp.prior.kernel, X, test_inputs, param)
        Kxx = gram(gp.prior.kernel, test_inputs, param)
        A = solve_triangular(L, Kfx.T, lower=True)
        latent_var = Kxx - jnp.sum(jnp.square(A), -2)
        latent_mean = jnp.matmul(A.T, nu)
        lvar = jnp.diag(latent_var)
        moment_fn = predictive_moments(gp.likelihood)
        pred_rv = moment_fn(latent_mean.ravel(), lvar)
        return pred_rv.variance()

    return variancef
