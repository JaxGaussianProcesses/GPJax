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

import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array, Float

from ..base import AbstractKernel
from ..computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)


class Polynomial(AbstractKernel):
    """The Polynomial kernel with variable degree."""

    def __init__(
        self,
        degree: int = 1,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Polynomial",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)
        self.degree = degree
        self.name = f"Polynomial Degree: {self.degree}"

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with shift parameter :math:`\\alpha` and variance :math:`\\sigma^2` through

        .. math::
            k(x, y) = \\Big( \\alpha + \\sigma^2 xy \\Big)^{d}

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x).squeeze()
        y = self.slice_input(y).squeeze()
        K = jnp.power(params["shift"] + jnp.dot(x * params["variance"], y), self.degree)
        return K.squeeze()

    def init_params(self, key: KeyArray) -> Dict:
        return {
            "shift": jnp.array([1.0]),
            "variance": jnp.array([1.0] * self.ndims),
        }
