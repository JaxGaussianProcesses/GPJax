from gpjax.objectives import marginal_ll
from gpjax import Prior
from gpjax.likelihoods import Bernoulli, Gaussian
from gpjax.kernels import RBF
from gpjax.parameters import transform, SoftplusTransformation, initialise
import jax.numpy as jnp
import jax.random as jr
import pytest
from typing import Callable
from tensorflow_probability.substrates.jax import distributions as tfd


def test_conjugate():
    posterior = Prior(kernel = RBF()) * Gaussian()
    mll = marginal_ll(posterior)
    assert isinstance(mll, Callable)
    neg_mll = marginal_ll(posterior, negative=True)
    x = jnp.linspace(-1., 1., 20).reshape(-1, 1)
    y = jnp.sin(x)
    params = transform(params=initialise(posterior), transformation=SoftplusTransformation)
    assert neg_mll(params, x, y) == jnp.array(-1.)*mll(params, x, y)


def test_non_conjugate():
    posterior = Prior(kernel = RBF()) * Bernoulli()
    mll = marginal_ll(posterior)
    assert isinstance(mll, Callable)
    neg_mll = marginal_ll(posterior, negative=True)
    n = 20
    x = jnp.linspace(-1., 1., n).reshape(-1, 1)
    y = jnp.sin(x)
    params = transform(params=initialise(posterior, n), transformation=SoftplusTransformation)
    assert neg_mll(params, x, y) == jnp.array(-1.)*mll(params, x, y)


def test_prior_mll():
    """
    Test that the MLL evaluation works with priors attached to the parameter values.
    """
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key, minval=-5.0, maxval=5.0, shape=(100, 1)), axis=0)
    f = lambda x: jnp.sin(jnp.pi * x) / (jnp.pi * x)
    y = f(x) + jr.normal(key, shape=x.shape) * 0.1
    posterior = Prior(kernel=RBF()) * Gaussian()
    mll = marginal_ll(posterior)

    params = initialise(posterior)
    priors = {
        "lengthscale": tfd.Gamma(1.0, 1.0),
        "variance": tfd.Gamma(2.0, 2.0),
        "obs_noise": tfd.Gamma(2.0, 2.0),
    }
    mll_eval = mll(params, x, y)
    mll_eval_priors = mll(params, x, y, priors)

    assert pytest.approx(mll_eval) == jnp.array(-115.72332969)
    assert pytest.approx(mll_eval_priors) == jnp.array(-118.97202259)