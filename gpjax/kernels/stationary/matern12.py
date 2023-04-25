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
from gpjax.typing import Array
from jaxtyping import Float
from beartype.typing import Union
from gpjax.typing import ScalarFloat

from ...base import param_field
from ..base import AbstractKernel
from .utils import build_student_t_distribution, euclidean_distance


@dataclass
class Matern12(AbstractKernel):
    """The Matérn kernel with smoothness parameter fixed at 0.5."""

    lengthscale: Union[ScalarFloat, Float[Array, "D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    name: str = "Matérn12"

    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> ScalarFloat:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\\ell` and variance :math:`\\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( -\\frac{\\lvert x-y \\rvert}{2\\ell^2}  \\Bigg)

        Args:
            x (Float[Array, "D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "D"]): The right hand argument of the kernel function's call
        Returns:
            ScalarFloat: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        K = self.variance * jnp.exp(-euclidean_distance(x, y))
        return K.squeeze()

    @property
    def spectral_density(self) -> tfd.Distribution:
        return build_student_t_distribution(nu=1)
