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
import functools as ft

import beartype.typing as tp
from flax.experimental import nnx
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)

from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.parameters import (
    Parameter,
    Real,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class AbstractKernel(nnx.Module):
    r"""Base kernel class.

    This class is the base class for all kernels in GPJax. It provides the basic
    functionality for evaluating a kernel function on a pair of inputs, as well as
    the ability to combine kernels using addition and multiplication.

    The class also provides a method for slicing the input matrix to select the
    relevant columns for the kernel's evaluation.

    Attributes:
        active_dims (tp.Union[list[int], slice]): The indices of the input dimensions
            that are active in the kernel's evaluation. If active_dims is a list, then
            the input to the kernel is indexed by the list, and n_dims
            is the length of the list. If active_dims is an integer, then the input to the
            kernel is not indexed, and n_dims is the value of the integer.
            If active_dims is a slice, then the input to the kernel is indexed by the slice,
            and n_dims is the length of the slice. Importantly, n_dims must always be
            inferable from active_dims.
        compute_engine (AbstractKernelComputation): The computation engine that is used to
            compute the kernel's cross-covariance and gram matrices.
        n_dims (int): The number of input dimensions of the kernel.
        name (str): The name of the kernel.
    """

    active_dims: tp.Union[list[int], slice]
    compute_engine: AbstractKernelComputation
    n_dims: int
    name: str = "AbstractKernel"

    def __init__(
        self,
        active_dims: tp.Union[list[int], int, slice],
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initialise the AbstractKernel class.

        Args:
            active_dims (tp.Union[list[int], int, slice]): The indices of the input dimensions
                that are active in the kernel's evaluation. If active_dims is a list, then
                the input to the kernel is indexed by the list, and the number of input dimensions
                is the length of the list. If active_dims is an integer, then the input to the
                kernel is not indexed, and the number of input dimensions is the value of the integer.
                If active_dims is a slice, then the input to the kernel is indexed by the slice,
                and the number of input dimensions is the length of the slice. Importantly, the number
                of active dimensions must be inferable from active_dims.
            compute_engine (AbstractKernelComputation): The computation engine that is used to
                compute the kernel's cross-covariance and gram matrices.
        """
        self.n_dims, self.active_dims = _check_active_dims(active_dims)
        self.compute_engine = compute_engine

    def cross_covariance(self, x: Num[Array, "N D"], y: Num[Array, "M D"]):
        return self.compute_engine.cross_covariance(self, x, y)

    def gram(self, x: Num[Array, "N D"]):
        return self.compute_engine.gram(self, x)

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
        x: Num[Array, " D"],
        y: Num[Array, " D"],
    ) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Num[Array, " D"]): The left hand input of the kernel function.
            y (Num[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        raise NotImplementedError

    def __add__(
        self, other: tp.Union["AbstractKernel", ScalarFloat]
    ) -> "AbstractKernel":
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

    def __radd__(
        self, other: tp.Union["AbstractKernel", ScalarFloat]
    ) -> "AbstractKernel":
        r"""Add two kernels together.
        Args:
            other (AbstractKernel): The kernel to be added to the current kernel.

        Returns
        -------
            AbstractKernel: A new kernel that is the sum of the two kernels.
        """
        return self.__add__(other)

    def __mul__(
        self, other: tp.Union["AbstractKernel", ScalarFloat]
    ) -> "AbstractKernel":
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


class Constant(AbstractKernel):
    r"""
    A constant kernel. This kernel evaluates to a constant for all inputs.
    The scalar value itself can be treated as a model hyperparameter and learned during training.
    """

    def __init__(
        self,
        active_dims: tp.Union[list[int], int, slice, None] = 1,
        constant: tp.Union[ScalarFloat, Parameter[ScalarFloat]] = jnp.array(0.0),
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        if isinstance(constant, Parameter):
            self.constant = constant
        else:
            self.constant = Real(jnp.array(constant))

        super().__init__(active_dims=active_dims, compute_engine=compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, " D"]): The left hand input of the kernel function.
            y (Float[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        return self.constant.value.squeeze()


class CombinationKernel(AbstractKernel):
    r"""A base class for products or sums of MeanFunctions."""

    def __init__(
        self,
        kernels: list[AbstractKernel],
        operator: tp.Callable,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        # Add kernels to a list, flattening out instances of this class therein, as in GPFlow kernels.
        kernels_list: list[AbstractKernel] = []
        for kernel in kernels:
            if not isinstance(kernel, AbstractKernel):
                raise TypeError("can only combine Kernel instances")  # pragma: no cover

            if isinstance(kernel, self.__class__):
                kernels_list.extend(kernel.kernels)
            else:
                kernels_list.append(kernel)

        self.kernels = kernels_list
        self.operator = operator

        active_dims = ft.reduce(lambda asum, x: asum + x.n_dims, kernels_list, 0)

        super().__init__(active_dims=active_dims, compute_engine=compute_engine)

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


@tp.overload
def _check_active_dims(active_dims: list[int]) -> tuple[int, list[int]]:
    ...


@tp.overload
def _check_active_dims(active_dims: int) -> tuple[int, slice]:  # noqa: F811
    ...


@tp.overload
def _check_active_dims(active_dims: slice) -> tuple[int, slice]:  # noqa: F811
    ...


def _check_active_dims(active_dims: tp.Union[list[int], int, slice]):  # noqa: F811
    if isinstance(active_dims, list):
        return len(active_dims), active_dims
    elif isinstance(active_dims, int):
        return active_dims, slice(None)
    elif isinstance(active_dims, slice):
        if active_dims.stop is None:
            raise ValueError("active_dims slice must have a stop value.")
        if active_dims.stop < 0:
            raise ValueError("active_dims slice stop value must be positive.")

        start = active_dims.start if active_dims.start is not None else 0
        step = active_dims.step if active_dims.step is not None else 1
        return (active_dims.stop - start) // step, active_dims
    else:
        raise TypeError(
            "Expected active_dims to be a list, int or slice."
            f" Got {type(active_dims)} instead."
        )


SumKernel = ft.partial(CombinationKernel, operator=jnp.sum)
ProductKernel = ft.partial(CombinationKernel, operator=jnp.prod)
