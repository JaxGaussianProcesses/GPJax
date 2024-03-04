# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
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
import typing as tp

import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax as tfp

from gpjax.typing import (
    Array,
    ScalarFloat,
)

tfd = tfp.distributions


def build_student_t_distribution(nu: int) -> tfd.Distribution:
    r"""Build a Student's t distribution with a fixed smoothness parameter.

    For a fixed half-integer smoothness parameter, compute the spectral density of a
    Matérn kernel; a Student's t distribution.

    Args:
        nu (int): The smoothness parameter of the Matérn kernel.

    Returns
    -------
        tfp.Distribution: A Student's t distribution with the same smoothness parameter.
    """
    dist = tfd.StudentT(df=nu, loc=0.0, scale=1.0)
    return dist


def squared_distance(x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
    r"""Compute the squared distance between a pair of inputs.

    Args:
        x (Float[Array, " D"]): First input.
        y (Float[Array, " D"]): Second input.

    Returns
    -------
        ScalarFloat: The squared distance between the inputs.
    """
    return jnp.sum((x - y) ** 2)


def euclidean_distance(x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
    r"""Compute the euclidean distance between a pair of inputs.

    Args:
        x (Float[Array, " D"]): First input.
        y (Float[Array, " D"]): Second input.

    Returns
    -------
        ScalarFloat: The euclidean distance between the inputs.
    """
    return jnp.sqrt(jnp.maximum(squared_distance(x, y), 1e-36))


# TODO: maybe improve the control flow here
def _check_lengthscale_dims_compat(
    lengthscale: tp.Union[ScalarFloat, Float[Array, " D"]], n_dims: tp.Union[int, None] 
) -> tp.Union[int, None]:
    r"""Check that the lengthscale parameter is compatible with the number of input dimensions.

    Args:
        lengthscale (Float[Array, " D"]): The lengthscale parameter.
        n_dims (int): The number of input dimensions.
    """
    ls_shape = jnp.shape(lengthscale)

    if ls_shape == ():
        return n_dims 
    else:
        if n_dims is None:
            return ls_shape[0]
        else:
            if ls_shape != (n_dims,):
                raise ValueError(
                    "Expected `lengthscale` to be compatible with the number " 
                    f"of input dimensions. Got `lengthscale` with shape {ls_shape}, "
                    f"but the number of input dimensions is {n_dims}."
                )
