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
from typing import List, Optional

import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array, Float

from ...parameters import Softplus, param_field
from ..base import AbstractKernel
from ..computations import DenseKernelComputation
from .utils import squared_distance


@dataclass
class RationalQuadratic(AbstractKernel):

    lengthscale: Float[Array, "D"] = param_field(jnp.array([1.0]), bijector=Softplus)
    variance: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=Softplus)
    alpha: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=Softplus)

    def __call__(self, x: jax.Array, y: jax.Array) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\\ell` and variance :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( 1 + \\frac{\\lVert x - y \\rVert^2_2}{2 \\alpha \\ell^2} \\Bigg)

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        K = self.variance * (1 + 0.5 * squared_distance(x, y) / self.alpha) ** (
            -self.alpha
        )
        return K.squeeze()
