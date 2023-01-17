# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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
from jaxtyping import Array, Float

from .base import AbstractKernel
from .computations import (
    AbstractKernelComputation,
    ConstantDiagonalKernelComputation,
    DenseKernelComputation,
    DiagonalKernelComputation,
)
from .utils import euclidean_distance, squared_distance


##########################################
# Euclidean kernels
##########################################
class Linear(AbstractKernel):
    """The linear kernel."""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Linear",
    ) -> None:
        super().__init__(
            compute_engine, active_dims, stationary, spectral, name
        )

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
        super().__init__(
            compute_engine, active_dims, stationary, spectral, name
        )
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
        K = jnp.power(
            params["shift"] + jnp.dot(x * params["variance"], y), self.degree
        )
        return K.squeeze()

    def init_params(self, key: KeyArray) -> Dict:
        return {
            "shift": jnp.array([1.0]),
            "variance": jnp.array([1.0] * self.ndims),
        }


class White(AbstractKernel, ConstantDiagonalKernelComputation):
    def __post_init__(self) -> None:
        super(White, self).__post_init__()

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\delta(x-y)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        K = jnp.all(jnp.equal(x, y)) * params["variance"]
        return K.squeeze()

    def init_params(self, key: Float[Array, "1 D"]) -> Dict:
        """Initialise the kernel parameters.

        Args:
            key (Float[Array, "1 D"]): The key to initialise the parameters with.

        Returns:
            Dict: The initialised parameters.
        """
        return {"variance": jnp.array([1.0])}


class Periodic(AbstractKernel):
    """The periodic kernel.

    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Periodic",
    ) -> None:
        super().__init__(
            compute_engine, active_dims, stationary, spectral, name
        )

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


__all__ = [
    "Linear",
    "Periodic",
    "Polynomial",
    "White",
]
