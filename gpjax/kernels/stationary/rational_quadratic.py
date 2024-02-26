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

from jaxtyping import Float

from gpjax.parameters import Parameter, PositiveReal
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.computations import AbstractKernelComputation, DenseKernelComputation
from gpjax.kernels.stationary.utils import squared_distance
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class RationalQuadratic(StationaryKernel):
    name: str = "Rational Quadratic"

    def __init__(
        self,
        active_dims: tp.Union[list[int], int, slice] = 1,
        lengthscale: tp.Union[ScalarFloat, Float[Array, " D"], Parameter] = 1.0,
        variance: tp.Union[ScalarFloat, Parameter] = 1.0,
        alpha: tp.Union[ScalarFloat, Parameter] = 1.0,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        if isinstance(alpha, Parameter):
            self.alpha = alpha
        else:
            self.alpha = PositiveReal(alpha)

        super().__init__(active_dims, lengthscale, variance, compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""Compute the Powered Exponential kernel between a pair of arrays.

        Evaluate the kernel on a pair of inputs $`(x, y)`$ with lengthscale parameter
        $`\ell`$ and variance $`\sigma^2`$.
        ```math
        k(x,y)=\sigma^2\exp\Bigg(1+\frac{\lVert x-y\rVert^2_2}{2\alpha\ell^2}\Bigg)
        ```

        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's call.

        Returns:
            ScalarFloat: The value of $`k(x, y)`$.
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        K = self.variance * (1 + 0.5 * squared_distance(x, y) / self.alpha) ** (
            -self.alpha
        )
        return K.squeeze()
