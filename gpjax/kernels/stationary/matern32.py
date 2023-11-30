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
from jaxtyping import Float
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
class Matern32(AbstractKernel):
    r"""The Matérn kernel with smoothness parameter fixed at 1.5."""

    lengthscale: Union[ScalarFloat, Float[Array, " D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    name: str = "Matérn32"

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
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        tau = euclidean_distance(x, y)
        K = self.variance * (1.0 + jnp.sqrt(3.0) * tau) * jnp.exp(-jnp.sqrt(3.0) * tau)
        return K.squeeze()

    @property
    def spectral_density(self) -> tfd.Distribution:
        return build_student_t_distribution(nu=3)
