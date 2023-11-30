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

import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.base import (
    param_field,
    static_field,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    ConstantDiagonalKernelComputation,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class White(AbstractKernel):
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    compute_engine: AbstractKernelComputation = static_field(
        ConstantDiagonalKernelComputation(), repr=False
    )
    name: str = "White"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""Compute the White noise kernel between a pair of arrays.

        Evaluate the kernel on a pair of inputs $`(x, y)`$ with variance $`\sigma^2`$:
        ```math
        k(x, y) = \sigma^2 \delta(x-y)
        ```

        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's call.

        Returns
        -------
            ScalarFloat: The value of $`k(x, y)`$.
        """
        K = jnp.all(jnp.equal(x, y)) * self.variance
        return K.squeeze()
