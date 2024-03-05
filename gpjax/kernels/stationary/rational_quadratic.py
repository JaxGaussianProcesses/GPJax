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
from flax.experimental import nnx
from jaxtyping import Float

from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.stationary.utils import squared_distance
from gpjax.parameters import PositiveReal
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class RationalQuadratic(StationaryKernel):
    name: str = "Rational Quadratic"

    def __init__(
        self,
        active_dims: tp.Union[list[int], int, slice],
        lengthscale: tp.Union[LengthscaleCompatible, nnx.Variable[Lengthscale]] = 1.0,
        variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        alpha: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        if isinstance(alpha, nnx.Variable):
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
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        K = self.variance.value * (
            1 + 0.5 * squared_distance(x, y) / self.alpha.value
        ) ** (-self.alpha.value)
        return K.squeeze()
