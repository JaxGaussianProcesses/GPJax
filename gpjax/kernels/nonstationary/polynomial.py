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
from gpjax.typing import (
    Array,
    ScalarFloat,
    ScalarInt,
)


@dataclass
class Polynomial(AbstractKernel):
    """The Polynomial kernel with variable degree."""

    degree: ScalarInt = static_field(2)
    shift: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())

    def __post_init__(self):
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
        K = jnp.power(self.shift + self.variance * jnp.dot(x, y), self.degree)
        return K.squeeze()
