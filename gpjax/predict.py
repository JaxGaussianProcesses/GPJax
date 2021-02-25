from .gps import Prior, ConjugatePosterior, NonConjugatePosterior
from .kernel import gram, cross_covariance
from .likelihoods import predictive_moments
from .types import Array
from .utils import I
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor, solve_triangular, cholesky
from multipledispatch import dispatch
from .types import Array


@dispatch(ConjugatePosterior, dict, jnp.DeviceArray, jnp.DeviceArray,
          jnp.DeviceArray)
def mean(gp: ConjugatePosterior, param: dict, test_inputs: Array,
         train_inputs: Array, train_outputs: Array):
    assert train_outputs.ndim == 2, f"2-dimensional training outputs are required. Current dimensional: {train_outputs.ndim}."
    ell, alpha = param['lengthscale'], param['variance']
    sigma = param['obs_noise']
    n_train = train_inputs.shape[0]

    Kff = alpha * gram(gp.prior.kernel, train_inputs / ell)
    Kfx = alpha * cross_covariance(gp.prior.kernel, train_inputs / ell,
                                   test_inputs / ell)

    prior_mean = gp.prior.mean_function(train_inputs)
    L = cho_factor(Kff + I(n_train) * sigma, lower=True)

    prior_distance = train_outputs - prior_mean
    weights = cho_solve(L, prior_distance)
    return jnp.dot(Kfx, weights)


@dispatch(ConjugatePosterior, dict, jnp.DeviceArray, jnp.DeviceArray,
          jnp.DeviceArray)
def variance(gp: ConjugatePosterior, param: dict, test_inputs: Array,
             train_inputs: Array, train_outputs: Array) -> Array:
    assert train_outputs.ndim == 2, f"2-dimensional training outputs are required. Current dimensional: {train_outputs.ndim}."
    ell, alpha = param['lengthscale'], param['variance']
    sigma = param['obs_noise']
    n_train = train_inputs.shape[0]
    n_test = test_inputs.shape[0]

    Kff = alpha * gram(gp.prior.kernel, train_inputs / ell)
    Kfx = alpha * cross_covariance(gp.prior.kernel, train_inputs / ell,
                                   test_inputs / ell)
    Kxx = alpha * gram(gp.prior.kernel, test_inputs / ell)

    L = cho_factor(Kff + I(n_train) * sigma, lower=True)
    latents = cho_solve(L, Kfx.T)
    return Kxx - jnp.dot(Kfx, latents)


@dispatch(NonConjugatePosterior, dict, jnp.DeviceArray, jnp.DeviceArray,
          jnp.DeviceArray)
def mean(gp: NonConjugatePosterior, param: dict, test_inputs: Array,
         train_inputs: Array, train_outputs: Array):
    ell, alpha, nu = param['lengthscale'], param['variance'], param['latent']
    n_train = train_inputs.shape[0]
    Kff = alpha * gram(gp.prior.kernel, train_inputs / ell)
    Kfx = alpha * cross_covariance(gp.prior.kernel, train_inputs / ell, test_inputs / ell)
    Kxx = alpha * gram(gp.prior.kernel, test_inputs / ell)
    L = jnp.linalg.cholesky(Kff + jnp.eye(train_inputs.shape[0]) * 1e-6)

    A = solve_triangular(L, Kfx.T, lower=True)
    latent_var = Kxx - jnp.sum(jnp.square(A), -2)
    latent_mean = jnp.matmul(A.T, nu)

    lvar = jnp.diag(latent_var)

    moment_fn = predictive_moments(gp.likelihood)
    pred_rv = moment_fn(latent_mean.ravel(), lvar)
    return pred_rv.mean()


@dispatch(NonConjugatePosterior, dict, jnp.DeviceArray, jnp.DeviceArray,
          jnp.DeviceArray)
def variance(gp: NonConjugatePosterior, param: dict, test_inputs: Array,
         train_inputs: Array, train_outputs: Array):
    ell, alpha, nu = param['lengthscale'], param['variance'], param['latent']
    n_train = train_inputs.shape[0]
    Kff = alpha * gram(gp.prior.kernel, train_inputs / ell)
    Kfx = alpha * cross_covariance(gp.prior.kernel, train_inputs / ell, test_inputs / ell)
    Kxx = alpha * gram(gp.prior.kernel, test_inputs / ell)
    L = jnp.linalg.cholesky(Kff + jnp.eye(train_inputs.shape[0]) * 1e-6)

    A = solve_triangular(L, Kfx.T, lower=True)
    latent_var = Kxx - jnp.sum(jnp.square(A), -2)
    latent_mean = jnp.matmul(A.T, nu)
    lvar = jnp.diag(latent_var)
    moment_fn = predictive_moments(gp.likelihood)
    pred_rv = moment_fn(latent_mean.ravel(), lvar)
    return pred_rv.variance()