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
from gpjax.kernels.stationary.utils import squared_distance
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class RBF(AbstractKernel):
    """The Radial Basis Function (RBF) kernel."""

    lengthscale: Union[ScalarFloat, Float[Array, "D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    name: str = "RBF"

    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> ScalarFloat:
        r"""Compute the RBF kernel between a pair of arrays.

        Evaluate the kernel on a pair of inputs$(x, y)$ with lengthscale parameter
        $\ell$ and variance $\sigma^2$:
        $$k(x,y)=\sigma^2\exp\Bigg(\frac{\lVert x - y \rVert^2_2}{2 \ell^2} \Bigg)$$

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, " D"]): The left hand argument of the kernel function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's call.

        Returns:
            ScalarFloat: The value of $k(x, y)$.
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        K = self.variance * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()

    @property
    def spectral_density(self) -> tfd.Normal:
        return tfd.Normal(loc=0.0, scale=1.0)
