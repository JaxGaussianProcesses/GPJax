# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

"""The number of Gauss-Hermite points to use for quadrature"""
DEFAULT_NUM_GAUSS_HERMITE_POINTS = 20


def gauss_hermite_quadrature(
    fun: Callable,
    mean: Float[Array, "N D"],
    sd: Float[Array, "N D"],
    deg: Optional[int] = DEFAULT_NUM_GAUSS_HERMITE_POINTS,
    *args,
    **kwargs
) -> Float[Array, "N"]:
    """
    Compute Gaussian-Hermite quadrature for a given function. The quadrature
    points are adjusted through the supplied mean and variance arrays.

    Args:
        fun (Callable): The function for which quadrature should be applied to.
        mean (Float[Array, "N D"]): The mean of the Gaussian distribution that
            is used to shift quadrature points.
        sd (Float[Array, "N D"]): The standard deviation of the Gaussian
            distribution that is used to scale quadrature points.
        deg (int, optional): The number of quadrature points that are to be used.
            Defaults to 20.

    Returns:
        Float[Array, "N"]: The evaluated integrals value.
    """
    gh_points, gh_weights = np.polynomial.hermite.hermgauss(deg)
    X = mean + jnp.sqrt(2.0) * sd * gh_points
    W = gh_weights / jnp.sqrt(jnp.pi)
    return jnp.sum(fun(X, *args, **kwargs) * W, axis=1)


__all__ = ["gauss_hermite_quadrature"]
