# Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
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
import jax.scipy as jsp
from jaxtyping import Float
from logbesselk.jax import log_bessel_k as log_kv
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.base import param_field
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.stationary.utils import (
    build_student_t_distribution,
    euclidean_distance,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class Matern(AbstractKernel):
    r"""The Matérn kernel with general smoothness parameter If you use smoothness 1/2, 3/2, 5/2, please use the corresponding Matern12, Matern32, Matern52 kernel for better efficiency.

    Or for smoothness approaching infinity, please use the RBF kernel.

    """

    smoothness: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    lengthscale: Union[ScalarFloat, Float[Array, " D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    name: str = "Matérn"

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        r"""Compute the Matérn 3/2 kernel between a pair of arrays.

        Evaluate the kernel on a pair of inputs $`(x, y)`$ with
        lengthscale parameter $`\ell`$ and variance $`\sigma^2`$.

        ```math
            k(x, y) = \sigma^2 \exp \Bigg(1+ \frac{\sqrt{3}\lvert x-y \rvert}{\ell^2}  \Bigg)\exp\Bigg(-\frac{\sqrt{3}\lvert x-y\rvert}{\ell^2} \Bigg)
        ```

        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's call.

        Returns
        -------
            ScalarFloat: The value of $k(x, y)$.
        """
        nu = self.smoothness
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        tau = euclidean_distance(x, y)
        weighted_distance = jnp.sqrt(2.0 * nu) * tau
        normalising_constant = (2.0 ** (1.0 - nu)) / jsp.special.gamma(nu)
        K = (
            self.variance
            * normalising_constant
            * (weighted_distance**nu)
            * jnp.exp(log_kv(nu, weighted_distance))
        )
        return K.squeeze()

    @property
    def spectral_density(self) -> tfd.Distribution:
        return build_student_t_distribution(nu=2.0 * self.smoothness)
