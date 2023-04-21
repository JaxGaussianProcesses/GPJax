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
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jaxtyping import Array, Float
from gpjax.utils import ScalarFloat

from ...base import param_field
from ..base import AbstractKernel


@dataclass
class Periodic(AbstractKernel):
    """The periodic kernel.

    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    lengthscale: Float[Array, "D"] = param_field(
        jnp.array([1.0]), bijector=tfb.Softplus()
    )
    variance: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    period: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    name: str = "Periodic"

    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> ScalarFloat:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\\ell` and variance :math:`\\sigma`

        TODO: update docstring

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( -0.5 \\sum_{i=1}^{d} \\Bigg)

        Args:
            x (Float[Array, "D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "D"]): The right hand argument of the kernel function's call
        Returns:
            ScalarFloat: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        sine_squared = (jnp.sin(jnp.pi * (x - y) / self.period) / self.lengthscale) ** 2
        K = self.variance * jnp.exp(-0.5 * jnp.sum(sine_squared, axis=0))
        return K.squeeze()
