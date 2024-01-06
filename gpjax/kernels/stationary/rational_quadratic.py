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
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.stationary.utils import squared_distance
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@nnx.dataclass
class RationalQuadratic(AbstractKernel):
    lengthscale: tp.Union[ScalarFloat, Float[Array, " D"]] = nnx.variable_field(
        nnx.Param, default=jnp.array(1.0)
    )
    variance: ScalarFloat = nnx.variable_field(nnx.Param, default=jnp.array(1.0))
    alpha: ScalarFloat = nnx.variable_field(nnx.Param, default=jnp.array(1.0))
    name: str = "Rational Quadratic"

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
