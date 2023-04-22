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
from jaxtyping import (
    Array,
    Float,
)
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.base import param_field
from gpjax.kernels.base import AbstractKernel


@dataclass
class Linear(AbstractKernel):
    """The linear kernel."""

    variance: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    name: str = "Linear"

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> Float[Array, "1"]:
        """Evaluate the linear kernel on a pair of inputs :math:`(x, y)` with variance parameter :math:`\\sigma`.

        .. math::
            k(x, y) = \\sigma^2 x^{T}y

        Args:
            x (Float[Array, "D"]): The left hand input of the kernel function.
            y (Float[Array, "D"]): The right hand input of the kernel function.

        Returns
        -------
            Float[Array, "1"]: The evaluated kernel function :math:`k(x, y)` at the supplied inputs.
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        K = self.variance * jnp.matmul(x.T, y)
        return K.squeeze()
