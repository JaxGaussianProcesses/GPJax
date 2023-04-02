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
import tensorflow_probability.substrates.jax.bijectors as tfb

from jaxtyping import Array, Float
from dataclasses import dataclass
from simple_pytree import static_field

from ..base import AbstractKernel
from ...base import param_field



@dataclass
class Polynomial(AbstractKernel):
    """The Polynomial kernel with variable degree."""

    degree: int = static_field(2)
    shift: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    variance: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())

    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with shift parameter :math:`\\alpha` and variance :math:`\\sigma^2` through

        .. math::
            k(x, y) = \\Big( \\alpha + \\sigma^2 xy \\Big)^{d}

        Args:
            x (Float[Array, "D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "D"]): The right hand argument of the kernel function's call

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x).squeeze()
        y = self.slice_input(y).squeeze()
        K = jnp.power(self.shift + jnp.dot(x * self.variance, y), self.degree)
        return K.squeeze()
