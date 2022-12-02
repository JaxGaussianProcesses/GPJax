# Copyright 2022 The JaxLinOp Contributors. All Rights Reserved.
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .diagonal_linear_operator import DiagonalLinearOperator

import abc
import jax.numpy as jnp
import distrax

from jaxtyping import Array, Float
from typing import Any, TypeVar, Iterable, Mapping, Generic, Tuple, Union

from . import pytree

# Generic type.
T = TypeVar("T")

# Generic nested type.
NestedT = Union[T, Iterable["NestedT"], Mapping[Any, "NestedT"]]

# Nested types.
ShapeT = TypeVar("ShapeT", bound=NestedT[Tuple[int, ...]])
DTypeT = TypeVar("DTypeT", bound=NestedT[jnp.dtype])


class LinearOperator(pytree.Pytree, Generic[ShapeT, DTypeT], metaclass=abc.ABCMeta):
    """Linear operator base class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise linear operator."""
        self._args = args
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Linear operator name."""
        return type(self).__name__

    @property
    @abc.abstractmethod
    def shape(self) -> ShapeT:
        """Linear operator shape."""
        raise NotImplementedError

    @property
    def dtype(self) -> DTypeT:
        """Linear operator data type."""
        return self._args[0].dtype

    @property
    def ndim(self) -> int:
        """Linear operator dimension."""
        return len(self.shape)

    @property
    def T(self) -> LinearOperator:
        """Transpose linear operator. Currently, we assume all linear operators are square and symmetric."""
        return self

    def __sub__(
        self, other: Union[LinearOperator, Float[Array, "N N"]]
    ) -> LinearOperator:
        """Subtract linear operator."""
        return self + (other * -1)

    def __rsub__(
        self, other: Union[LinearOperator, Float[Array, "N N"]]
    ) -> LinearOperator:
        """Reimplimentation of subtract linear operator."""
        return (self * -1) + other

    def __add__(
        self, other: Union[LinearOperator, Float[Array, "N N"]]
    ) -> LinearOperator:
        """Add linear operator."""
        raise NotImplementedError

    def __radd__(
        self, other: Union[LinearOperator, Float[Array, "N N"]]
    ) -> LinearOperator:
        """Reimplimentation of add linear operator."""
        return self + other

    @abc.abstractmethod
    def __mul__(self, other: float) -> LinearOperator:
        """Multiply linear operator by scalar."""
        raise NotImplementedError

    def __rmul__(self, other: float) -> LinearOperator:
        """Reimplimentation of multiply linear operator by scalar."""
        return self.__mul__(other)

    @abc.abstractmethod
    def _add_diagonal(self, other: DiagonalLinearOperator) -> LinearOperator:
        """Add diagonal linear operator to a linear operator, useful e.g., for adding jitter."""
        return NotImplementedError

    @abc.abstractmethod
    def __matmul__(
        self, other: Union[LinearOperator, Float[Array, "N M"]]
    ) -> Union[LinearOperator, Float[Array, "N M"]]:
        """Matrix multiplication."""
        raise NotImplementedError

    def __rmatmul__(
        self, other: Union[LinearOperator, Float[Array, "N M"]]
    ) -> Union[LinearOperator, Float[Array, "N M"]]:
        """Reimplimentation of matrix multiplication."""
        # Exploit the fact that linear operators are square and symmetric.
        if other.ndim == 1:
            return self.T @ other
        return (self.T @ other.T).T

    @abc.abstractmethod
    def diagonal(self) -> Float[Array, "N"]:
        """Diagonal of the linear operator.

        Returns:
            Float[Array, "N"]: Diagonal of the linear operator.
        """

        raise NotImplementedError

    def trace(self) -> Float[Array, "1"]:
        """Trace of the linear matrix.

        Returns:
            Float[Array, "1"]: Trace of the linear matrix.
        """
        return jnp.sum(self.diagonal())

    def log_det(self) -> Float[Array, "1"]:
        """Log determinant of the linear matrix. Default implementation uses dense Cholesky decomposition.

        Returns:
            Float[Array, "1"]: Log determinant of the linear matrix.
        """

        root = self.to_root()

        return 2.0 * jnp.sum(jnp.log(root.diagonal()))

    def to_root(self) -> LinearOperator:
        """Compute the root of the linear operator via the Cholesky decomposition.

        Returns:
            Float[Array, "N N"]: Lower Cholesky decomposition of the linear operator.
        """

        from .triangular_linear_operator import LowerTriangularLinearOperator

        L = jnp.linalg.cholesky(self.to_dense())

        return LowerTriangularLinearOperator.from_dense(L)

    def inverse(self) -> LinearOperator:
        """Inverse of the linear matrix. Default implementation uses dense Cholesky decomposition.

        Returns:
            LinearOperator: Inverse of the linear matrix.
        """

        from .dense_linear_operator import DenseLinearOperator

        n = self.shape[0]

        return DenseLinearOperator(self.solve(jnp.eye(n)))

    def solve(self, rhs: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Solve linear system. Default implementation uses dense Cholesky decomposition.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns:
            Float[Array, "N M"]: Solution of the linear system.
        """

        root = self.to_root()
        rootT = root.T

        return rootT.solve(root.solve(rhs))

    @abc.abstractmethod
    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covaraince matrix from the linear operator.

        Returns:
            Float[Array, "N N"]: Dense linear matrix.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> LinearOperator:
        """Construct linear operator from dense matrix.

        Args:
            dense (Float[Array, "N N"]): Dense matrix.

        Returns:
            LinearOperator: Linear operator.
        """
        raise NotImplementedError

    @classmethod
    def to_bijector(cls) -> distrax.Bijector:
        """Construct bijector from linear operator.

        Returns:
            Bijector: Bijector.
        """
        raise NotImplementedError


__all__ = [
    "LinearOperator",
]
