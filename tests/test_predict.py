import jax.numpy as jnp
import jax.random as jr

from gpjax import Prior
from gpjax.kernels import RBF
from gpjax.likelihoods import Bernoulli, Gaussian
from gpjax.parameters import initialise
from gpjax.predict import mean, variance


def test_conjugate_mean():
    key = jr.PRNGKey(123)
    x = jr.uniform(key, shape=(20, 1), minval=-3.0, maxval=3.0)
    y = jnp.sin(x)

    posterior = Prior(kernel=RBF()) * Gaussian()
    params = initialise(posterior)

    xtest = jnp.linspace(-3.0, 3.0, 30).reshape(-1, 1)
    mu = mean(posterior, params, xtest, x, y)
    assert mu.shape == (xtest.shape[0], y.shape[1])


def test_conjugate_variance():
    key = jr.PRNGKey(123)
    x = jr.uniform(key, shape=(20, 1), minval=-3.0, maxval=3.0)
    y = jnp.sin(x)

    posterior = Prior(kernel=RBF()) * Gaussian()
    params = initialise(posterior)

    xtest = jnp.linspace(-3.0, 3.0, 30).reshape(-1, 1)
    sigma = variance(posterior, params, xtest, x, y)
    assert sigma.shape == (xtest.shape[0], xtest.shape[0])


def test_non_conjugate_mean():
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key, shape=(10, 1), minval=-1.0, maxval=1.0), axis=0)
    y = 0.5 * jnp.sign(jnp.cos(3 * x + jr.normal(key, shape=x.shape) * 0.05)) + 0.5
    xtest = jnp.linspace(-1.05, 1.05, 50).reshape(-1, 1)

    posterior = Prior(kernel=RBF()) * Bernoulli()
    params = initialise(posterior, x.shape[0])

    mu = mean(posterior, params, xtest, x, y)
    assert mu.shape == (xtest.shape[0],)


def test_non_conjugate_variance():
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key, shape=(10, 1), minval=-1.0, maxval=1.0), axis=0)
    y = 0.5 * jnp.sign(jnp.cos(3 * x + jr.normal(key, shape=x.shape) * 0.05)) + 0.5
    xtest = jnp.linspace(-1.05, 1.05, 50).reshape(-1, 1)

    posterior = Prior(kernel=RBF()) * Bernoulli()
    params = initialise(posterior, x.shape[0])

    sigma = variance(posterior, params, xtest, x, y)
    assert sigma.shape == (xtest.shape[0],)
