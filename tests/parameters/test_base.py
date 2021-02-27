from gpjax.parameters.base import initialise, _initialise_hyperparams, complete
from gpjax import Prior
from gpjax.kernels import RBF
from gpjax.mean_functions import Zero
from gpjax.likelihoods import Gaussian, Bernoulli
import jax.numpy as jnp
import pytest


def test_complete():
    posterior = Prior(kernel = RBF()) * Gaussian()
    partial_params = {'lengthscale': jnp.array(1.0)}
    full_params = complete(partial_params, posterior)
    assert list(full_params.keys()) == ['lengthscale', 'variance', 'obs_noise']


def test_initialise():
    posterior = Prior(kernel = RBF()) * Gaussian()
    params = initialise(posterior)
    assert list(params.keys()) == ['lengthscale', 'variance', 'obs_noise']


@pytest.mark.parametrize('n', [1, 10])
def test_non_conjugate_initialise(n):
    posterior = Prior(kernel=RBF()) * Bernoulli()
    params = initialise(posterior, n)
    assert list(params.keys()) == ['lengthscale', 'variance', 'latent']
    assert params['latent'].shape == (n, 1)


def test_hyperparametr_initialise():
    params = _initialise_hyperparams(RBF(), Zero())
    assert list(params.keys()) == ['lengthscale', 'variance']