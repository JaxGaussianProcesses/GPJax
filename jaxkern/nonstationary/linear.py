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


##########################################
# Euclidean kernels
##########################################
class Linear(AbstractKernel):
    """The linear kernel."""

    def __init__(
        self,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        name: Optional[str] = "Linear",
    ) -> None:
        super().__init__(
            DenseKernelComputation,
            active_dims,
            spectral_density=None,
            name=name,
        )
        self._stationary = False

    def __call__(self, params: dict, x: jax.Array, y: jax.Array) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance parameter :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 x^{T}y

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        K = params["variance"] * jnp.matmul(x.T, y)
        return K.squeeze()

    def init_params(self, key: KeyArray) -> Dict:
        return {"variance": jnp.array([1.0])}
