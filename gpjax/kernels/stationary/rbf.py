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
import tensorflow_probability.substrates.jax as tfp

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import AbstractKernelComputation, DenseKernelComputation
from gpjax.kernels.stationary.utils import squared_distance, _check_lengthscale_dims_compat
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class RBF(AbstractKernel):
    r"""The Radial Basis Function (RBF) kernel."""

    name: str = "RBF"

    def __init__(
        self,
        active_dims: tp.Union[list[int], int, slice],
        lengthscale: tp.Union[ScalarFloat, Float[Array, " D"]] = 1.0,
        variance: ScalarFloat = 1.0,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        super().__init__(active_dims=active_dims, compute_engine=compute_engine)

        _check_lengthscale_dims_compat(lengthscale, self.n_dims)

        self.lengthscale = lengthscale
        self.variance = variance


    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""Compute the RBF kernel between a pair of arrays.

        Evaluate the kernel on a pair of inputs $`(x, y)`$ with lengthscale parameter
        $`\ell`$ and variance $`\sigma^2`$:
        ```math
        k(x,y)=\sigma^2\exp\Bigg(- \frac{\lVert x - y \rVert^2_2}{2 \ell^2} \Bigg)
        ```

        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's call.

        Returns:
            ScalarFloat: The value of $`k(x, y)`$.
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        K = self.variance * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)