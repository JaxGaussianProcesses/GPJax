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


import abc
from dataclasses import dataclass
from functools import partial

from beartype.typing import (
    Callable,
    List,
    Optional,
    Type,
    Union,
)
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class AbstractKernel(Module):
    r"""Base kernel class."""

    compute_engine: Type[AbstractKernelComputation] = static_field(
        DenseKernelComputation
    )
    active_dims: Optional[List[int]] = static_field(None)
    name: str = static_field("AbstractKernel")

    @property
    def ndims(self):
        return 1 if not self.active_dims else len(self.active_dims)

    def cross_covariance(self, x: Num[Array, "N D"], y: Num[Array, "M D"]):
        return self.compute_engine(self).cross_covariance(x, y)

    def gram(self, x: Num[Array, "N D"]):
        return self.compute_engine(self).gram(x)

    def slice_input(self, x: Float[Array, "... D"]) -> Float[Array, "... Q"]:
        r"""Slice out the relevant columns of the input matrix.

        Select the relevant columns of the supplied matrix to be used within the
        kernel's evaluation.

        Args:
            x (Float[Array, "... D"]): The matrix or vector that is to be sliced.

        Returns
        -------
            Float[Array, "... Q"]: A sliced form of the input matrix.
        """
        return x[..., self.active_dims] if self.active_dims is not None else x

    @abc.abstractmethod
    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, " D"]): The left hand input of the kernel function.
            y (Float[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        raise NotImplementedError

    def __add__(self, other: Union["AbstractKernel", ScalarFloat]) -> "AbstractKernel":
        r"""Add two kernels together.
        Args:
            other (AbstractKernel): The kernel to be added to the current kernel.

        Returns
        -------
            AbstractKernel: A new kernel that is the sum of the two kernels.
        """
        if isinstance(other, AbstractKernel):
            return SumKernel(kernels=[self, other])
        else:
            return SumKernel(kernels=[self, Constant(other)])

    def __radd__(self, other: Union["AbstractKernel", ScalarFloat]) -> "AbstractKernel":
        r"""Add two kernels together.
        Args:
            other (AbstractKernel): The kernel to be added to the current kernel.

        Returns
        -------
            AbstractKernel: A new kernel that is the sum of the two kernels.
        """
        return self.__add__(other)

    def __mul__(self, other: Union["AbstractKernel", ScalarFloat]) -> "AbstractKernel":
        r"""Multiply two kernels together.

        Args:
            other (AbstractKernel): The kernel to be multiplied with the current kernel.

        Returns
        -------
            AbstractKernel: A new kernel that is the product of the two kernels.
        """
        if isinstance(other, AbstractKernel):
            return ProductKernel(kernels=[self, other])
        else:
            return ProductKernel(kernels=[self, Constant(other)])

    @property
    def spectral_density(self) -> Optional[tfd.Distribution]:
        return None


@dataclass
class Constant(AbstractKernel):
    r"""
    A constant kernel. This kernel evaluates to a constant for all inputs.
    The scalar value itself can be treated as a model hyperparameter and learned during training.
    """

    constant: ScalarFloat = param_field(jnp.array(0.0))

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, " D"]): The left hand input of the kernel function.
            y (Float[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        return self.constant.squeeze()


@dataclass
class CombinationKernel(AbstractKernel):
    r"""A base class for products or sums of MeanFunctions."""

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
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, " D"]): The left hand input of the kernel function.
            y (Float[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        return self.operator(jnp.stack([k(x, y) for k in self.kernels]))


SumKernel = partial(CombinationKernel, operator=jnp.sum)
ProductKernel = partial(CombinationKernel, operator=jnp.prod)
