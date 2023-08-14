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
import jax
import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.bijectors as tfb

import gpjax as gpx
from gpjax.base.param import param_field
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class ChangePointRBF(gpx.kernels.AbstractKernel):
    r"""The change point kernel for two RBFs."""

    variance1: ScalarFloat = param_field(
        jnp.array(1.0), bijector=tfb.Softplus(), trainable=True
    )
    lengthscale1: Union[ScalarFloat, Float[Array, " D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus(), trainable=True
    )
    variance2: ScalarFloat = param_field(
        jnp.array(1.0), bijector=tfb.Softplus(), trainable=True
    )
    lengthscale2: Union[ScalarFloat, Float[Array, " D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus(), trainable=True
    )
    tswitch: float = param_field(
        jnp.float64(1.0), bijector=tfb.Identity(), trainable=False
    )
    name: str = "ChangePoint"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""Compute the changepoint kernel for two RBFs between a pair of arrays.
        ```

        Args:
            x (Float[Array, " N"]): The left hand argument of the kernel function's call.
            y (Float[Array, " N"]): The right hand argument of the kernel function's call.
            self.tswitch: The point at which to change to a different kernel.
            self.variance1: Variance parameter for first RBF (if x, y both less than tswitch)
            self.lengthscale1: Lengthscale parameter for first RBF (if x, y both less than tswitch)
            self.variance2: Variance parameter for second RBF (if x, y both greater than or equal to tswitch)
            self.lengthscale2: Lengthscale parameter for second RBF (if x, y both greater than or equal to tswitch)

        Returns:
            ScalarFloat: The value of $`k(x, y)`. Three possible scenarios:
            (1) RBF with parameters variance1, lengthscale1 if both x and y are less than tswitch
            (2) RBF with parameters variance2, lengthscale2 is both x and y are greater than or equal to tswitch
            (3) 0 otherwise$.
        """

        def k11(x1, x2, v1=1, l1=0.1, v2=2, l2=0.2):
            r"""Compute $`k(x, y)`$ using RBF with parameters variance1 and lengthscale1"""
            out = v1 * jnp.exp(-(((x1 - x2) / l1) ** 2))
            return out.squeeze()

        def k22(x1, x2, v1=1, l1=0.1, v2=2, l2=0.2):
            r"""Compute $`k(x, y)`$ using RBF with parameters variance2 and lengthscale2"""
            out = v2 * jnp.exp(-(((x1 - x2) / l2) ** 2))
            return out.squeeze()

        def k12(x1, x2, v1=1, l1=0.1, v2=2, l2=0.2):
            r"""Return 0 covariance"""
            out = jnp.float64(0)
            return out.squeeze()

        def k21(x1, x2, v1=1, l1=0.1, v2=2, l2=0.2):
            r"""Return 0 covariance"""
            out = jnp.float64(0)
            return out.squeeze()

        def GetFunctionIndex(x, y, tswitch=1):
            r"""
            Specify four possible indices given x, y, and tswitch.

            Args:
                    x: Left hand argument of kernel function's call
                    y: Right hand argument of kernel function's call
                    tswitch: point at which to change to a different kernel
            """
            # Four possible indexes: 0, 1, 2, 3
            indx = 3  # if indx doesn't get set to 0, 1, or 2, then default 3
            # lessthan means that at tswitch, you are already switched
            cond1 = jnp.less(x, tswitch)
            cond2 = jnp.less(y, tswitch)
            indx = jnp.where(jnp.logical_and(cond1, cond2), 0, indx)
            indx = jnp.where(jnp.logical_and(jnp.invert(cond1), cond2), 1, indx)
            indx = jnp.where(jnp.logical_and(cond1, jnp.invert(cond2)), 2, indx)
            return indx.squeeze().astype("uint8")

        indx = GetFunctionIndex(x, y, tswitch=self.tswitch)
        flst = [k11, k21, k12, k22]
        K = jax.lax.switch(
            indx,
            flst,
            x,
            y,
            self.variance1,
            self.lengthscale1,
            self.variance2,
            self.lengthscale2,
        )
        return K.squeeze()
