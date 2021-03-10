from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gpjax import Prior
from gpjax.kernels import RBF, to_spectral
from gpjax.likelihoods import Bernoulli, Gaussian
from gpjax.objectives import marginal_ll
from gpjax.parameters import SoftplusTransformation, initialise, transform


def test_conjugate():
    key = jr.PRNGKey(123)
    kern = to_spectral(RBF(), 10)
    posterior = Prior(kernel=kern) * Gaussian()
    mll = marginal_ll(posterior)
    assert isinstance(mll, Callable)
    neg_mll = marginal_ll(posterior, negative=True)
    x = jnp.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = jnp.sin(x)
    params = transform(params=initialise(key, posterior), transformation=SoftplusTransformation)
    assert neg_mll(params, x, y) == jnp.array(-1.0) * mll(params, x, y)
    nmll = neg_mll(params, x, y)
    assert nmll.shape == ()