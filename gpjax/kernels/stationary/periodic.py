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

from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array

from ..base import AbstractKernel
from ..computations import (
    DenseKernelComputation,
)


class Periodic(AbstractKernel):
    """The periodic kernel.

    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    def __init__(
        self,
        active_dims: Optional[List[int]] = None,
        name: Optional[str] = "Periodic",
    ) -> None:
        super().__init__(
            DenseKernelComputation, active_dims, spectral_density=None, name=name
        )
        self._stationary = True

    def __call__(self, params: dict, x: jax.Array, y: jax.Array) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\\ell` and variance :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( -0.5 \\sum_{i=1}^{d} \\Bigg)

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        sine_squared = (
            jnp.sin(jnp.pi * (x - y) / params["period"]) / params["lengthscale"]
        ) ** 2
        K = params["variance"] * jnp.exp(-0.5 * jnp.sum(sine_squared, axis=0))
        return K.squeeze()

    def init_params(self, key: KeyArray) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "period": jnp.array([1.0] * self.ndims),
        }
