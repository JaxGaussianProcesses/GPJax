import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax import Prior
from gpjax.kernels import RBF, to_spectral
from gpjax.likelihoods import Bernoulli, Gaussian, Poisson
from gpjax.mean_functions import Zero
from gpjax.parameters.base import _initialise_hyperparams, complete, initialise


def test_complete():
    posterior = Prior(kernel=RBF()) * Gaussian()
    partial_params = {"lengthscale": jnp.array(1.0)}
    full_params = complete(partial_params, posterior)
    assert list(full_params.keys()) == sorted(["lengthscale", "variance", "obs_noise"])


def test_initialise():
    posterior = Prior(kernel=RBF()) * Gaussian()
    params = initialise(posterior)
    assert list(params.keys()) == sorted(["lengthscale", "variance", "obs_noise"])


@pytest.mark.parametrize("n", [1, 10])
def test_non_conjugate_initialise(n):
    posterior = Prior(kernel=RBF()) * Bernoulli()
    params = initialise(posterior, n)
    assert list(params.keys()) == sorted(["lengthscale", "variance", "latent"])
    assert params["latent"].shape == (n, 1)


def test_hyperparametr_initialise():
    params = _initialise_hyperparams(RBF(), Zero())
    assert list(params.keys()) == sorted(["lengthscale", "variance"])


@pytest.mark.parametrize("lik", [Bernoulli, Poisson, Gaussian])
def test_dtype(lik):
    posterior = Prior(kernel=RBF()) * lik()
    for k, v in initialise(posterior, 10).items():
        assert v.dtype == jnp.float64


def test_spectral():
    key = jr.PRNGKey(123)
    kernel = to_spectral(RBF(), 10)
    posterior = Prior(kernel=kernel) * Gaussian()
    params = initialise(key, posterior)
    assert list(params.keys()) == sorted(["basis_fns", "obs_noise", "lengthscale", "variance"])
    assert params["basis_fns"].shape == (10, 1)
