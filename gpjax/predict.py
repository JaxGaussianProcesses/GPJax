from .gps import Prior, ExactPosterior
from .kernel import gram, cross_covariance
from .types import Array
from .utils import I
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor
from multipledispatch import dispatch
from .types import Array


@dispatch(ExactPosterior, dict, jnp.DeviceArray, jnp.DeviceArray,
          jnp.DeviceArray)
def mean(gp: ExactPosterior, param: dict, test_inputs: Array,
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


@dispatch(ExactPosterior, dict, jnp.DeviceArray, jnp.DeviceArray,
          jnp.DeviceArray)
def variance(gp: ExactPosterior, param: dict, test_inputs: Array,
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
