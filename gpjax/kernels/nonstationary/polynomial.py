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

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from jaxtyping import Array, Float
from simple_pytree import static_field
from gpjax.utils import ScalarFloat

from ...base import param_field
from ..base import AbstractKernel


@dataclass
class Polynomial(AbstractKernel):
    """The Polynomial kernel with variable degree."""

    degree: int = static_field(2)
    shift: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())

    def __post_init__(self):
        self.name = f"Polynomial (degree {self.degree})"

    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> ScalarFloat:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with shift parameter
        :math:`\\alpha` and variance :math:`\\sigma^2` through

        .. math::
            k(x, y) = \\Big( \\alpha + \\sigma^2 xy \\Big)^{d}

        Args:
            x (Float[Array, "D"]): The left hand argument of the kernel function's
                call.
            y (Float[Array, "D"]): The right hand argument of the kernel function's
                call

        Returns:
            ScalarFloat: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        K = jnp.power(self.shift + self.variance * jnp.dot(x, y), self.degree)
        return K.squeeze()
