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
from beartype.typing import Union

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from gpjax.typing import Array
from jaxtyping import Float
from gpjax.typing import ScalarFloat

from ...base import param_field
from ..base import AbstractKernel
from .utils import euclidean_distance


@dataclass
class PoweredExponential(AbstractKernel):
    """The powered exponential family of kernels.

    Key reference is Diggle and Ribeiro (2007) - "Model-based Geostatistics".

    """

    lengthscale: Union[ScalarFloat, Float[Array, "D"]] = param_field(
        jnp.array([1.0]), bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    power: ScalarFloat = param_field(jnp.array(1.0))
    name: str = "Powered Exponential"

    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> ScalarFloat:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\\ell`, :math:`\\sigma` and power :math:`\\kappa`.

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( - \\Big( \\frac{\\lVert x - y \\rVert^2}{\\ell^2} \\Big)^\\kappa \\Bigg)

        Args:
            x (Float[Array, "D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "D"]): The right hand argument of the kernel function's call

        Returns:
            ScalarFloat: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        K = self.variance * jnp.exp(-euclidean_distance(x, y) ** self.power)
        return K.squeeze()
