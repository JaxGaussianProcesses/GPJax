import jax.numpy as jnp
import jax.random as jr
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gpjax import Prior
from gpjax.kernels import RBF, to_spectral
from gpjax.likelihoods import Gaussian
from gpjax.parameters import initialise
from gpjax.sampling import random_variable, sample


@pytest.mark.parametrize("n", [1, 10])
def test_prior_random_variable(n):
    f = Prior(kernel=RBF())
    sample_points = jnp.linspace(-1.0, 1.0, num=n).reshape(-1, 1)
    params = initialise(RBF())
    rv = random_variable(f, params, sample_points)
    assert isinstance(rv, tfd.MultivariateNormalFullCovariance)


@pytest.mark.parametrize("n", [1, 10])
def test_posterior_random_variable(n):
    f = Prior(kernel=RBF()) * Gaussian()
    x = jnp.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    y = jnp.sin(x)
    sample_points = jnp.linspace(-1.0, 1.0, num=n).reshape(-1, 1)
    params = initialise(f)
    rv = random_variable(f, params, sample_points, x, y)
    assert isinstance(rv, tfd.MultivariateNormalFullCovariance)


@pytest.mark.parametrize("n_sample", [1, 10])
@pytest.mark.parametrize("n", [1, 10])
def test_prior_sample(n, n_sample):
    key = jr.PRNGKey(123)
    f = Prior(kernel=RBF())
    sample_points = jnp.linspace(-1.0, 1.0, num=n).reshape(-1, 1)
    params = initialise(RBF())
    samples = sample(key, f, params, sample_points, n_samples=n_sample)
    assert samples.shape == (n_sample, sample_points.shape[0])


@pytest.mark.parametrize("n_sample", [1, 10])
@pytest.mark.parametrize("n", [1, 10])
def test_posterior_sample(n, n_sample):
    key = jr.PRNGKey(123)
    f = Prior(kernel=RBF()) * Gaussian()
    x = jnp.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    y = jnp.sin(x)
    sample_points = jnp.linspace(-1.0, 1.0, num=n).reshape(-1, 1)
    params = initialise(f)
    rv = random_variable(f, params, sample_points, x, y)
    samples = sample(key, rv, n_samples=n_sample)
    assert samples.shape == (n_sample, sample_points.shape[0])


def test_spectral_sample():
    key = jr.PRNGKey(123)
    M = 10
    x = jnp.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = jnp.sin(x)
    sample_points = jnp.linspace(-1.0, 1.0, num=50).reshape(-1, 1)
    kernel = to_spectral(RBF(), M)
    post = Prior(kernel=kernel) * Gaussian()
    params = initialise(key, post)
    sparams = {"basis_fns": params["basis_fns"]}
    del params["basis_fns"]
    posterior_rv = random_variable(post, params, x, y, sample_points, static_params=sparams)
    assert isinstance(posterior_rv, tfd.Distribution)
    assert isinstance(posterior_rv, tfd.MultivariateNormalFullCovariance)
