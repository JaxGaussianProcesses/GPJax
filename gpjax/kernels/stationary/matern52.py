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
from typing import Dict, List, Optional

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jaxtyping import Array, Float

from ...parameters import Softplus, param_field
from ..base import AbstractKernel
from ..computations import DenseKernelComputation
from .utils import build_student_t_distribution, euclidean_distance


@dataclass
class Matern52(AbstractKernel):
    """The Matérn kernel with smoothness parameter fixed at 2.5."""

    lengthscale: Float[Array, "D"] = param_field(jnp.array([1.0]), bijector=Softplus)
    variance: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=Softplus)

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\\ell` and variance :math:`\\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{5}\\lvert x-y \\rvert}{\\ell^2} + \\frac{5\\lvert x - y \\rvert^2}{3\\ell^2} \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{5}\\lvert x-y\\rvert}{\\ell^2} \\Bigg)

        Args:
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        tau = euclidean_distance(x, y)
        K = (
            self.variance
            * (1.0 + jnp.sqrt(5.0) * tau + 5.0 / 3.0 * jnp.square(tau))
            * jnp.exp(-jnp.sqrt(5.0) * tau)
        )
        return K.squeeze()

    @property
    def spectral_density(self):
        return build_student_t_distribution(nu=5)
