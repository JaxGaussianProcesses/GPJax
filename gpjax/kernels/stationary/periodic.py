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

import beartype.typing as tp

import jax.numpy as jnp
from jaxtyping import Float

from gpjax.parameters import Parameter, PositiveReal
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.computations import AbstractKernelComputation, DenseKernelComputation
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class Periodic(StationaryKernel):
    r"""The periodic kernel.

    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    name: str = "Periodic"

    def __init__(
        self,
        active_dims: tp.Union[list[int], int, slice] = 1,
        lengthscale: tp.Union[ScalarFloat, Float[Array, " D"], Parameter] = 1.0,
        variance: tp.Union[ScalarFloat, Parameter] = 1.0,
        period: tp.Union[ScalarFloat, Parameter] = 1.0,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        
        if isinstance(period, Parameter):
            self.period = period
        else:
            self.period = PositiveReal(period)

        super().__init__(active_dims, lengthscale, variance, compute_engine)

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        r"""Compute the Periodic kernel between a pair of arrays.

        Evaluate the kernel on a pair of inputs $`(x, y)`$ with length-scale parameter $`\ell`$, variance $`\sigma^2`$
        and period $`p`$.
        ```math
        k(x, y) = \sigma^2 \exp \left( -\frac{1}{2} \sum_{i=1}^{D} \left(\frac{\sin (\pi (x_i - y_i)/p)}{\ell}\right)^2 \right)
        ```

        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's call
        Returns:
            ScalarFloat: The value of $`k(x, y)`$.
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        sine_squared = (jnp.sin(jnp.pi * (x - y) / self.period) / self.lengthscale) ** 2
        K = self.variance * jnp.exp(-0.5 * jnp.sum(sine_squared, axis=0))
        return K.squeeze()
