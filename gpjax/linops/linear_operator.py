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


import abc
from dataclasses import dataclass

from beartype.typing import (
    Any,
    Generic,
    Iterable,
    Mapping,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import jax.numpy as jnp
from jaxtyping import Float
from simple_pytree import (
    Pytree,
    static_field,
)

from gpjax.typing import (
    Array,
    ScalarFloat,
)

# Generic type.
T = TypeVar("T")

# Generic nested type.
NestedT = Union[T, Iterable["NestedT"], Mapping[Any, "NestedT"]]

# Nested types.
DTypes = Union[Type[jnp.float32], Type[jnp.float64], Type[jnp.int32], Type[jnp.int64]]
ShapeT = TypeVar("ShapeT", bound=NestedT[Tuple[int, ...]])
DTypeT = TypeVar("DTypeT", bound=NestedT[DTypes])

# The Generic type is used for type checking the LinearOperator's shape and datatype.
# `static_field` is used to mark nodes of the PyTree that don't change under JAX transformations.
# this is important, so that we e.g., don't take the gradient with respect to the shape!


@dataclass
class LinearOperator(Pytree, Generic[ShapeT, DTypeT]):
    """Linear operator base class."""

    shape: ShapeT = static_field()
    dtype: DTypeT = static_field()

    def __repr__(self) -> str:
        """Linear operator representation."""
        return f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype.__name__})"

    @property
    def ndim(self) -> int:
        """Linear operator dimension."""
        return len(self.shape)

    @property
    def T(self) -> "LinearOperator":
        """Transpose linear operator. Currently, we assume all linear operators are square and symmetric."""
        return self

    def __sub__(
        self, other: Union["LinearOperator", Float[Array, "N N"]]
    ) -> "LinearOperator":
        """Subtract linear operator."""
        return self + (other * -1)

    def __rsub__(
        self, other: Union["LinearOperator", Float[Array, "N N"]]
    ) -> "LinearOperator":
        """Reimplimentation of subtract linear operator."""
        return (self * -1) + other

    def __add__(
        self, other: Union["LinearOperator", Float[Array, "N N"]]
    ) -> "LinearOperator":
        """Add linear operator."""
        raise NotImplementedError

    def __radd__(
        self, other: Union["LinearOperator", Float[Array, "N N"]]
    ) -> "LinearOperator":
        """Reimplimentation of add linear operator."""
        return self + other

    @abc.abstractmethod
    def __mul__(self, other: ScalarFloat) -> "LinearOperator":
        """Multiply linear operator by scalar."""
        raise NotImplementedError

    def __rmul__(self, other: ScalarFloat) -> "LinearOperator":
        """Reimplimentation of multiply linear operator by scalar."""
        return self.__mul__(other)

    @abc.abstractmethod
    def _add_diagonal(
        self,
        other: "gpjax.linops.diagonal_linear_operator.DiagonalLinearOperator",  # noqa: F821
    ) -> "LinearOperator":
        """Add diagonal linear operator to a linear operator, useful e.g., for adding jitter."""
        return NotImplementedError

    @abc.abstractmethod
    def __matmul__(
        self, other: Union["LinearOperator", Float[Array, "N M"]]
    ) -> Union["LinearOperator", Float[Array, "N M"]]:
        """Matrix multiplication."""
        raise NotImplementedError

    def __rmatmul__(
        self, other: Union["LinearOperator", Float[Array, "N M"]]
    ) -> Union["LinearOperator", Float[Array, "N M"]]:
        """Reimplimentation of matrix multiplication."""
        # Exploit the fact that linear operators are square and symmetric.
        if other.ndim == 1:
            return self.T @ other
        return (self.T @ other.T).T

    @abc.abstractmethod
    def diagonal(self) -> Float[Array, " N"]:
        """Diagonal of the linear operator.

        Returns
        -------
            Float[Array, " N"]: Diagonal of the linear operator.
        """
        raise NotImplementedError

    def trace(self) -> ScalarFloat:
        """Trace of the linear matrix.

        Returns
        -------
            ScalarFloat: Trace of the linear matrix.
        """
        return jnp.sum(self.diagonal())

    def log_det(self) -> ScalarFloat:
        """Log determinant of the linear matrix. Default implementation uses dense Cholesky decomposition.

        Returns
        -------
            ScalarFloat: Log determinant of the linear matrix.
        """
        root = self.to_root()

        return 2.0 * jnp.sum(jnp.log(root.diagonal()))

    def to_root(self) -> "LinearOperator":
        """Compute the root of the linear operator via the Cholesky decomposition.

        Returns
        -------
            Float[Array, "N N"]: Lower Cholesky decomposition of the linear operator.
        """
        from gpjax.linops.triangular_linear_operator import (
            LowerTriangularLinearOperator,
        )

        L = jnp.linalg.cholesky(self.to_dense())

        return LowerTriangularLinearOperator.from_dense(L)

    def inverse(self) -> "LinearOperator":
        """Inverse of the linear matrix. Default implementation uses dense Cholesky decomposition.

        Returns
        -------
            LinearOperator: Inverse of the linear matrix.
        """
        from gpjax.linops.dense_linear_operator import DenseLinearOperator

        n = self.shape[0]

        return DenseLinearOperator(self.solve(jnp.eye(n)))

    def solve(self, rhs: Float[Array, "... M"]) -> Float[Array, "... M"]:
        """Solve linear system. Default implementation uses dense Cholesky decomposition.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns
        -------
            Float[Array, "N M"]: Solution of the linear system.
        """
        root = self.to_root()
        rootT = root.T

        return rootT.solve(root.solve(rhs))

    @abc.abstractmethod
    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense matrix from the linear operator.

        Returns
        -------
            Float[Array, "N N"]: Dense linear matrix.
        """
        raise NotImplementedError

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> "LinearOperator":
        """Construct linear operator from dense matrix.

        Args:
            dense (Float[Array, "N N"]): Dense matrix.

        Returns
        -------
            LinearOperator: Linear operator.
        """
        raise NotImplementedError


__all__ = [
    "LinearOperator",
]
