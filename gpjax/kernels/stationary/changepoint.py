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

from beartype.typing import (
    Callable,
    List,
)
import jax
import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.base import (
    param_field,
    static_field,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class ChangePoint(AbstractKernel):
    r"""A change point kernel

    See Saatci, Turner, Rasmussen 2010 ICML paper for details.

    self.kernels: A list of exactly two kernels that will be switched.
    self.tswitch: The point at which to change to a different kernel.
        for example: if x and y are both less than tswitch, then you would use kernels[0]
                     if x and y are both greater than or equal to tswitch, then you would use kernels[1]
                     otherwise return cross-covariance of 0

    """

    kernels: List[AbstractKernel] = None
    operator: Callable = static_field(None)
    tswitch: ScalarFloat = param_field(
        jnp.array(1.0), bijector=tfb.Identity(), trainable=False
    )

    def __post_init__(self):
        # Add kernels to a list, flattening out instances of this class therein, as in GPFlow kernels.
        kernels_list: List[AbstractKernel] = []

        for kernel in self.kernels:
            if not isinstance(kernel, AbstractKernel):
                raise TypeError("can only combine Kernel instances")  # pragma: no cover

            if isinstance(kernel, self.__class__):
                kernels_list.extend(kernel.kernels)
            else:
                kernels_list.append(kernel)

        self.kernels = kernels_list

    def __call__(
        self,
        x: Float[Array, " 1"],
        y: Float[Array, " 1"],
    ) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, " 1"]): The left hand input of the kernel function.
            y (Float[Array, " 1"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """

        def get_function_index(x, y, tswitch):
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

        def k_zero(x, y):
            r"""Return 0 covariance"""
            out = jnp.float64(0)
            return out.squeeze()

        indx = get_function_index(x, y, tswitch=self.tswitch)

        flst = [self.kernels[0].__call__, k_zero, k_zero, self.kernels[1].__call__]
        K = jax.lax.switch(indx, flst, x, y)

        return K.squeeze()
