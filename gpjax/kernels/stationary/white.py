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
import typing as tp

from flax import nnx
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.computations import (
    AbstractKernelComputation,
    ConstantDiagonalKernelComputation,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)


class White(StationaryKernel):
    r"""The White noise kernel.

    Computes the covariance for pairs of inputs $(x, y)$ with variance $\sigma^2$:
    $$
    k(x, y) = \sigma^2 \delta(x-y)
    $$
    """

    name: str = "White"

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = ConstantDiagonalKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            variance: the variance of the kernel σ.
            n_dims: The number of input dimensions.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix
        """
        super().__init__(active_dims, 1.0, variance, n_dims, compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        K = jnp.all(jnp.equal(x, y)) * self.variance.value
        return K.squeeze()
