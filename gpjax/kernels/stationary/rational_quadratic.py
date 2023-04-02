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
import tensorflow_probability.substrates.jax.distributions as tfd

from jaxtyping import Array, Float
from dataclasses import dataclass

from ...base import param_field
from ..base import AbstractKernel
from .utils import squared_distance


@dataclass
class RationalQuadratic(AbstractKernel):

    lengthscale: Float[Array, "D"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    variance: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    alpha: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())

    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\\ell` and variance :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( 1 + \\frac{\\lVert x - y \\rVert^2_2}{2 \\alpha \\ell^2} \\Bigg)

        Args:
            x (Float[Array, "D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "D"]): The right hand argument of the kernel function's call
        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        K = self.variance * (1 + 0.5 * squared_distance(x, y) / self.alpha) ** (
            -self.alpha
        )
        return K.squeeze()
