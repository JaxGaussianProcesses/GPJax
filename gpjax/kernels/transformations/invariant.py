#Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
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
import abc
from dataclasses import dataclass
from functools import partial
from beartype.typing import Callable, Optional
import jax.numpy as jnp
from jaxtyping import Float, Int

from gpjax.base import (
    param_field,
    static_field,
)
from gpjax.kernels import AbstractKernel
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class AbstractGroup:
    r"""Base class for Groups"""
    @abc.abstractmethod
    def orbit(self, x: Float[Array, "D"]) -> Float[Array, "M D"]:
        r"""Compute the orbit of a point under the group action."""
        raise NotImplementedError

    
@dataclass
class FiniteGroup(AbstractGroup):
    r"""todo"""
    orbit_fn: Callable[[Float[Array, "D"]], Float[Array, "M D"]] = static_field(None)
    def orbit(self, x: Float[Array, "D"]) -> Float[Array, "M D"]:
        r"""Compute the orbit of a point under the group action."""
        return  self.orbit_fn(x)# [N * n_group_elements, D]
    

@dataclass
class GroupInvariantKernel(AbstractKernel):
    r"""todo add fundamental domain"""
    base_kernel: AbstractKernel = None
    group: AbstractGroup = static_field(None)
    operator: Callable = static_field(None)

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs. todo

        Args:
            x (Float[Array, " D"]): The left hand input of the kernel function.
            y (Float[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        x_orbit, y_orbit = self.group.orbit(x), self.group.orbit(y)
        k_xy = self.base_kernel.cross_covariance(x_orbit, y_orbit) # [n_group_elements * n_goup_elements]
        return self.operator(k_xy)

SumGroupInvariantKernel = partial(GroupInvariantKernel, operator=jnp.sum)
ProductGroupInvariantKernel = partial(GroupInvariantKernel, operator=jnp.prod)
