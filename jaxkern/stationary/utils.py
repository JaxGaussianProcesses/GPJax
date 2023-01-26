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

import jax.numpy as jnp
from jaxtyping import Array, Float
import distrax as dx
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


def build_student_t_distribution(nu: int) -> dx.Distribution:
    """For a fixed half-integer smoothness parameter, compute the spectral density of a Matérn kernel; a Student's t distribution.

    Args:
        nu (int): The smoothness parameter of the Matérn kernel.

    Returns:
        dx.Distribution: A Student's t distribution with the same smoothness parameter.
    """
    tfp_dist = tfd.StudentT(df=nu, loc=0.0, scale=1.0)
    return dx._src.distributions.distribution_from_tfp.distribution_from_tfp(tfp_dist)


def squared_distance(
    x: Float[Array, "1 D"], y: Float[Array, "1 D"]
) -> Float[Array, "1"]:
    """Compute the squared distance between a pair of inputs.

    Args:
        x (Float[Array, "1 D"]): First input.
        y (Float[Array, "1 D"]): Second input.

    Returns:
        Float[Array, "1"]: The squared distance between the inputs.
    """

    return jnp.sum((x - y) ** 2)


def euclidean_distance(
    x: Float[Array, "1 D"], y: Float[Array, "1 D"]
) -> Float[Array, "1"]:
    """Compute the euclidean distance between a pair of inputs.

    Args:
        x (Float[Array, "1 D"]): First input.
        y (Float[Array, "1 D"]): Second input.

    Returns:
        Float[Array, "1"]: The euclidean distance between the inputs.
    """

    return jnp.sqrt(jnp.maximum(squared_distance(x, y), 1e-36))
