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
    """Compute Gaussian-Hermite quadrature for a given function. The quadrature points are adjusted through the supplied mean and variance arrays.

    Args:
        fun (Callable): The function for which quadrature should be applied to.
        mean (Array): The mean of the Gaussian distribution that is used to shift quadrature points.
        var (Array): The variance of the Gaussian distribution that is used to scale quadrature points.
        deg (int, optional): The number of quadrature points that are to be used. Defaults to 20.

    Returns:
        Array: The evaluated integrals value.
    """
    gh_points, gh_weights = np.polynomial.hermite.hermgauss(deg)
    stdev = jnp.sqrt(var)
    X = mean + jnp.sqrt(2.0) * stdev * gh_points
    W = gh_weights / jnp.sqrt(jnp.pi)
    return jnp.sum(fun(X, *args, **kwargs) * W, axis=1)
