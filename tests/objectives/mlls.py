from gpjax.objectives import marginal_ll
from gpjax import Prior
from gpjax.likelihoods import Bernoulli, Gaussian
from gpjax.kernels import RBF
from gpjax.parameters import transform, SoftplusTransformation, initialise
import jax.numpy as jnp
import pytest
from typing import Callable


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
