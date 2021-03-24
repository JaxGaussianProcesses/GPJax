from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gpjax import Dataset, Prior
from gpjax.config import get_defaults
from gpjax.kernels import RBF
from gpjax.likelihoods import Bernoulli, Gaussian
from gpjax.objectives import marginal_ll
from gpjax.parameters import build_all_transforms, build_unconstrain, initialise


def test_conjugate():
    posterior = Prior(kernel=RBF()) * Gaussian()

    x = jnp.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = jnp.sin(x)
    D = Dataset(X=x, y=y)
    params = initialise(posterior)
    config = get_defaults()
    unconstrainer, constrainer = build_all_transforms(params.keys(), config)
    params = unconstrainer(params)
    mll = marginal_ll(posterior, transform=constrainer)
    assert isinstance(mll, Callable)
    neg_mll = marginal_ll(posterior, transform=constrainer, negative=True)
    assert neg_mll(params, D) == jnp.array(-1.0) * mll(params, D)


def test_non_conjugate():
    posterior = Prior(kernel=RBF()) * Bernoulli()
    n = 20
    x = jnp.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = jnp.sin(x)
    D = Dataset(X=x, y=y)
    params = initialise(posterior, 20)
    config = get_defaults()
    unconstrainer, constrainer = build_all_transforms(params.keys(), config)
    params = unconstrainer(params)
    mll = marginal_ll(posterior, transform=constrainer)
    assert isinstance(mll, Callable)
    neg_mll = marginal_ll(posterior, transform=constrainer, negative=True)
    assert neg_mll(params, D) == jnp.array(-1.0) * mll(params, D)


def test_prior_mll():
    """
    Test that the MLL evaluation works with priors attached to the parameter values.
    """
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key, minval=-5.0, maxval=5.0, shape=(100, 1)), axis=0)
    f = lambda x: jnp.sin(jnp.pi * x) / (jnp.pi * x)
    y = f(x) + jr.normal(key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    posterior = Prior(kernel=RBF()) * Gaussian()

    params = initialise(posterior)
    config = get_defaults()
    constrainer, unconstrainer = build_all_transforms(params.keys(), config)
    params = unconstrainer(params)
    print(params)

    mll = marginal_ll(posterior, transform=constrainer)

    priors = {
        "lengthscale": tfd.Gamma(1.0, 1.0),
        "variance": tfd.Gamma(2.0, 2.0),
        "obs_noise": tfd.Gamma(2.0, 2.0),
    }
    mll_eval = mll(params, D)
    mll_eval_priors = mll(params, D, priors)

    assert pytest.approx(mll_eval) == jnp.array(-103.28180663)
    assert pytest.approx(mll_eval_priors) == jnp.array(-105.509218857)
