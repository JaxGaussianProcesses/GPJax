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

from gpjax.base import param_field
from gpjax.kernels.base import AbstractKernel
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class Linear(AbstractKernel):
    r"""The linear kernel."""

    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    name: str = "Linear"

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        r"""Compute the linear kernel between a pair of arrays.

        For a pair of inputs $`x, y \in \mathbb{R}^{D}`$, let's evaluate the linear
        kernel $`k(x, y)=\sigma^2 x^{\top}y`$ where $`\sigma^\in \mathbb{R}_{>0}`$ is the
        kernel's variance parameter.

        Args:
            x (Float[Array, " D"]): The left hand input of the kernel function.
            y (Float[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function $`k(x, y)`$ at the supplied inputs.
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        K = self.variance * jnp.matmul(x.T, y)
        return K.squeeze()
