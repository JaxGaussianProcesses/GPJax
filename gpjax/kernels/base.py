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

from __future__ import annotations

import abc
import jax.numpy as jnp
from typing import List, Callable, Union
from jaxtyping import Array, Float
from functools import partial
from ..parameters import Module, param_field
from simple_pytree import static_field
from dataclasses import dataclass
from functools import partial

from .computations import AbstractKernelComputation, DenseKernelComputation


@dataclass
class AbstractKernel(Module):
    """Base kernel class."""

    compute_engine: AbstractKernelComputation = static_field(DenseKernelComputation)
    active_dims: List[int] = static_field(None)

    @property
    def ndims(self):
        return 1 if not self.active_dims else len(self.active_dims)

    def cross_covariance(self, x: Float[Array, "N D"], y: Float[Array, "M D"]):
        return self.compute_engine(self).cross_covariance(x, y)

    def gram(self, x: Float[Array, "N D"]):
        return self.compute_engine(self).gram(x)

    def slice_input(self, x: Float[Array, "N D"]) -> Float[Array, "N S"]:
        """Select the relevant columns of the supplied matrix to be used within the kernel's evaluation.

        Args:
            x (Float[Array, "N D"]): The matrix or vector that is to be sliced.
        Returns:
            Float[Array, "N S"]: A sliced form of the input matrix.
        """
        return x[..., self.active_dims]

    @abc.abstractmethod
    def __call__(
        self,
        x: Float[Array, "D"],
        y: Float[Array, "D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, "D"]): The left hand input of the kernel function.
            y (Float[Array, "D"]): The right hand input of the kernel function.

        Returns:
            Float[Array, "1"]: The evaluated kernel function at the supplied inputs.
        """
        raise NotImplementedError

    def __add__(
        self, other: Union[AbstractKernel, Float[Array, "1"]]
    ) -> AbstractKernel:
        """Add two kernels together.
        Args:
            other (AbstractKernel): The kernel to be added to the current kernel.

        Returns:
            AbstractKernel: A new kernel that is the sum of the two kernels.
        """

        if isinstance(other, AbstractKernel):
            return SumKernel([self, other])

        return SumKernel([self, Constant(other)])

    def __radd__(
        self, other: Union[AbstractKernel, Float[Array, "1"]]
    ) -> AbstractKernel:
        """Add two kernels together.
        Args:
            other (AbstractKernel): The kernel to be added to the current kernel.

        Returns:
            AbstractKernel: A new kernel that is the sum of the two kernels.
        """
        return self.__add__(other)

    def __mul__(
        self, other: Union[AbstractKernel, Float[Array, "1"]]
    ) -> AbstractKernel:
        """Multiply two kernels together.

        Args:
            other (AbstractKernel): The kernel to be multiplied with the current kernel.

        Returns:
            AbstractKernel: A new kernel that is the product of the two kernels.
        """
        if isinstance(other, AbstractKernel):
            return ProductKernel([self, other])

        return ProductKernel([self, Constant(other)])


@dataclass
class Constant(AbstractKernel):
    """
    A constant mean function. This function returns a repeated scalar value for all inputs.
    The scalar value itself can be treated as a model hyperparameter and learned during training.
    """

    constant: Float[Array, "1"] = param_field(jnp.array(0.0))

    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, "D"]): The left hand input of the kernel function.
            y (Float[Array, "D"]): The right hand input of the kernel function.

        Returns:
            Float[Array, "1"]: The evaluated kernel function at the supplied inputs.
        """
        return self.constant.squeeze()


@dataclass
class CombinationKernel(AbstractKernel):
    """A base class for products or sums of MeanFunctions."""

    kernels: List[AbstractKernel] = None
    operator: Callable = static_field(None)

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
        x: Float[Array, "D"],
        y: Float[Array, "D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, "D"]): The left hand input of the kernel function.
            y (Float[Array, "D"]): The right hand input of the kernel function.

        Returns:
            Float[Array, "1"]: The evaluated kernel function at the supplied inputs.
        """
        return self.operator(jnp.stack([k(x, y) for k in self.kernels]))


SumKernel = partial(CombinationKernel, operator=jnp.sum)
ProductKernel = partial(CombinationKernel, operator=jnp.sum)
