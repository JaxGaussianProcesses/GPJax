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
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.parameters import PositiveReal
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)


class Polynomial(AbstractKernel):
    """The Polynomial kernel with variable degree."""

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        degree: int = 2,
        shift: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 0.0,
        variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        super().__init__(active_dims, n_dims, compute_engine)

        self.degree = degree

        if isinstance(shift, nnx.Variable):
            self.shift = shift
        else:
            self.shift = PositiveReal(shift)
            if tp.TYPE_CHECKING:
                self.shift = tp.cast(PositiveReal[ScalarArray], self.shift)

        if isinstance(variance, nnx.Variable):
            self.variance = variance
        else:
            self.variance = PositiveReal(variance)
            if tp.TYPE_CHECKING:
                self.variance = tp.cast(PositiveReal[ScalarArray], self.variance)

        self.name = f"Polynomial (degree {self.degree})"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""Compute the polynomial kernel of degree $`d`$ between a pair of arrays.

        For a pair of inputs $`x, y \in \mathbb{R}^{D}`$, let's evaluate the polynomial
        kernel $`k(x, y)=\left( \alpha + \sigma^2 x y\right)^{d}`$ where
        $`\sigma^\in \mathbb{R}_{>0}`$ is the kernel's variance parameter, shift
        parameter $`\alpha`$ and integer degree $`d`$.

        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel function's
                call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's
                call

        Returns
        -------
            ScalarFloat: The value of $`k(x, y)`$.
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        K = jnp.power(
            self.shift.value + self.variance.value * jnp.dot(x, y), self.degree
        )
        return K.squeeze()
