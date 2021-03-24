from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gpjax import Dataset, Prior
from gpjax.config import get_defaults
from gpjax.kernels import RBF, to_spectral
from gpjax.likelihoods import Bernoulli, Gaussian
from gpjax.objectives import marginal_ll
from gpjax.parameters import build_all_transforms, initialise


def test_spectral():
    key = jr.PRNGKey(123)
    kern = to_spectral(RBF(), 10)
    posterior = Prior(kernel=kern) * Gaussian()
    x = jnp.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = jnp.sin(x)
    D = Dataset(X=x, y=y)
    params = initialise(key, posterior)
    config = get_defaults()
    unconstrainer, constrainer = build_all_transforms(params.keys(), config)
    params = unconstrainer(params)
    mll = marginal_ll(posterior, transform=constrainer)
    assert isinstance(mll, Callable)
    neg_mll = marginal_ll(posterior, transform=constrainer, negative=True)
    assert neg_mll(params, D) == jnp.array(-1.0) * mll(params, D)
    nmll = neg_mll(params, D)
    assert nmll.shape == ()
