from typing import Callable

import jax.numpy as jnp
import numpy as np

from .types import Array

"""The number of Gauss-Hermite points to use for quadrature"""
DEFAULT_NUM_GAUSS_HERMITE_POINTS = 20


def gauss_hermite_quadrature(
    fun: Callable,
    mean: Array,
    var: Array,
    deg: int = DEFAULT_NUM_GAUSS_HERMITE_POINTS,
    *args,
    **kwargs
) -> Array:
    gh_points, gh_weights = np.polynomial.hermite.hermgauss(deg)
    stdev = jnp.sqrt(var)
    X = mean + jnp.sqrt(2.0) * stdev * gh_points
    W = gh_weights / jnp.sqrt(jnp.pi)
    return jnp.sum(fun(X, *args, **kwargs) * W, axis=1)
