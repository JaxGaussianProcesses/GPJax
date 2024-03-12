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
        active_dims: The indices of the input dimensions
            that are active in the kernel's evaluation, represented by a list of integers
            or a slice object.
        compute_engine: The computation engine that is used to
            compute the kernel's cross-covariance and gram matrices.
        n_dims: The number of input dimensions of the kernel.
        name: The name of the kernel.
    """

    active_dims: tp.Union[list[int], slice] = slice(None)
    compute_engine: AbstractKernelComputation
    n_dims: tp.Union[int, None]
    name: str = "AbstractKernel"

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initialise the AbstractKernel class.

        Args:
            active_dims: the indices of the input dimensions
                that are active in the kernel's evaluation, represented by a list of
                integers or a slice object. Defaults to a full slice.
            n_dims: the number of input dimensions of the kernel.
            compute_engine: the computation engine that is used to compute the kernel's
                cross-covariance and gram matrices. Defaults to DenseKernelComputation.
        """

        active_dims = active_dims or slice(None)

        _check_active_dims(active_dims)
        _check_n_dims(n_dims)

        self.active_dims, self.n_dims = _check_dims_compat(active_dims, n_dims)

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
        active_dims: tp.Union[list[int], slice, None] = None,
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

        super().__init__(compute_engine=compute_engine)

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


def _check_active_dims(active_dims: tp.Any):
    if not isinstance(active_dims, (list, slice)):
        raise TypeError(
            f"Expected active_dims to be a list or slice. Got {active_dims} instead."
        )


def _check_n_dims(n_dims: tp.Any):
    if not isinstance(n_dims, (int, type(None))):
        raise TypeError(
            "Expected n_dims to be an integer or None (unspecified)."
            f" Got {n_dims} instead."
        )


def _check_dims_compat(
    active_dims: tp.Union[list[int], slice],
    n_dims: tp.Union[int, None],
):
    err = ValueError(
        "Expected the length of active_dims to be equal to the specified n_dims."
        f" Got {active_dims} active dimensions and {n_dims} input dimensions."
    )

    if isinstance(active_dims, list) and isinstance(n_dims, int):
        if len(active_dims) != n_dims:
            raise err

    if isinstance(active_dims, slice) and isinstance(n_dims, int):
        start = active_dims.start or 0
        stop = active_dims.stop or n_dims
        step = active_dims.step or 1
        if len(range(start, stop, step)) != n_dims:
            raise err

    if isinstance(active_dims, list) and n_dims is None:
        n_dims = len(active_dims)

    if isinstance(active_dims, slice) and n_dims is None:
        if active_dims == slice(None):
            pass
        else:
            start = active_dims.start or 0
            stop = active_dims.stop or n_dims
            step = active_dims.step or 1
            n_dims = len(range(start, stop, step))

    return active_dims, n_dims


SumKernel = ft.partial(CombinationKernel, operator=jnp.sum)
ProductKernel = ft.partial(CombinationKernel, operator=jnp.prod)
