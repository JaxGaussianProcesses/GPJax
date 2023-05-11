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

from dataclasses import dataclass

from beartype.typing import Union
import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.base import (
    param_field,
    static_field,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.typing import (
    Array,
    ScalarFloat,
    ScalarInt,
)


@dataclass
class ArcCosine(AbstractKernel):
    r"""The ArCosine kernel.

    This kernel is non-stationary and resembles the behavior of neural networks.
    See Section 3.1 of
    [Cho and Saul (2011)](https://arxiv.org/abs/1112.3712) for
    additional details.
    """

    order: ScalarInt = static_field(0)
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    weight_variance: Union[ScalarFloat, Float[Array, " D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    bias_variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())

    def __post_init__(self):
        if self.order not in [0, 1, 2]:
            raise ValueError("ArcCosine kernel only implemented for orders 0, 1 and 2.")

        self.name = f"ArcCosine (order {self.order})"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs $`(x, y)`$

        Args:
            x (Float[Array, "D"]): The left hand argument of the kernel function's
                call.
            y (Float[Array, "D"]): The right hand argument of the kernel function's
                call

        Returns
        -------
            ScalarFloat: The value of $`k(x, y)`$.
        """

        x = self.slice_input(x)
        y = self.slice_input(y)

        x_x = self._weighted_prod(x, x)
        x_y = self._weighted_prod(x, y)
        y_y = self._weighted_prod(y, y)

        cos_theta = x_y / jnp.sqrt(x_x * y_y)
        jitter = 1e-15  # improve numerical stability
        theta = jnp.arccos(jitter + (1 - 2 * jitter) * cos_theta)

        K = self._J(theta)
        K *= jnp.sqrt(x_x) ** self.order
        K *= jnp.sqrt(y_y) ** self.order
        K *= self.variance / jnp.pi

        return K.squeeze()

    def _weighted_prod(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> ScalarFloat:
        r"""Calculate the weighted product between two arguments.

        Args:
            x (Float[Array, "D"]): The left hand argument.
            y (Float[Array, "D"]): The right hand argument.
        Returns
        -------
            ScalarFloat: The value of the weighted product between the two arguments``.
        """
        return jnp.inner(self.weight_variance * x, y) + self.bias_variance

    def _J(self, theta: ScalarFloat) -> ScalarFloat:
        r"""Evaluate the angular dependency function corresponding to the desired order.

        Args:
            theta (Float[Array, "1"]): The weighted angle between inputs.

        Returns
        -------
            Float[Array, "1"]: The value of the angular dependency function`.
        """

        if self.order == 0:
            return jnp.pi - theta
        elif self.order == 1:
            return jnp.sin(theta) + (jnp.pi - theta) * jnp.cos(theta)
        else:
            return 3.0 * jnp.sin(theta) * jnp.cos(theta) + (jnp.pi - theta) * (
                1.0 + 2.0 * jnp.cos(theta) ** 2
            )
