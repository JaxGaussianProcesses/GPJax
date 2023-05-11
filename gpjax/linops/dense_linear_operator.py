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


from dataclasses import dataclass

from beartype.typing import Union
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.linops.linear_operator import LinearOperator
from gpjax.linops.utils import to_linear_operator
from gpjax.typing import (
    Array,
    ScalarFloat,
    VecNOrMatNM,
)


def _check_matrix(matrix: Array) -> None:
    if matrix.ndim != 2:
        raise ValueError(
            "The `matrix` must have at two dimensions, but "
            f"`scale.shape = {matrix.shape}`."
        )

    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"The `matrix` must be a square matrix, but `scale.shape = {matrix.shape}`."
        )


@dataclass
class DenseLinearOperator(LinearOperator):
    """Dense covariance operator."""

    matrix: Float[Array, "N N"]

    def __init__(self, matrix: Float[Array, "N N"], dtype: jnp.dtype = None) -> None:
        """Initialize the covariance operator.

        Args:
            matrix (Float[Array, "N N"]): Dense matrix.
        """
        _check_matrix(matrix)

        if dtype is not None:
            matrix = matrix.astype(dtype)

        self.matrix = matrix
        self.shape = matrix.shape
        self.dtype = matrix.dtype

    def __add__(
        self, other: Union[LinearOperator, Float[Array, "N N"]]
    ) -> LinearOperator:
        """Add diagonal to another linear operator.

        Args:
            other (Union[LinearOperator, Float[Array, "N N"]]): Other linear operator. Dimension of both operators must match. If the other linear operator is not a DiagonalLinearOperator, dense matrix addition is used.

        Returns
        -------
            LinearOperator: linear operator plus the diagonal linear operator.
        """
        from gpjax.linops.diagonal_linear_operator import DiagonalLinearOperator
        from gpjax.linops.zero_linear_operator import ZeroLinearOperator

        other = to_linear_operator(other)

        if isinstance(other, DiagonalLinearOperator):
            return self._add_diagonal(other)

        elif isinstance(other, DenseLinearOperator):
            return DenseLinearOperator(matrix=self.matrix + other.matrix)

        elif isinstance(other, ZeroLinearOperator):
            return self

        else:
            raise NotImplementedError

    def __mul__(self, other: ScalarFloat) -> LinearOperator:
        """Multiply covariance operator by scalar.

        Args:
            other (LinearOperator): Scalar.

        Returns
        -------
            LinearOperator: Covariance operator multiplied by a scalar.
        """
        return DenseLinearOperator(matrix=self.matrix * other)

    def _add_diagonal(
        self,
        other: "gpjax.linops.diagonal_linear_operator.DiagonalLinearOperator",  # noqa: F821
    ) -> LinearOperator:
        """Add diagonal to the covariance operator,  useful for computing, Kxx + Io².

        Args:
            other (DiagonalLinearOperator): Diagonal covariance operator to add to the covariance operator.

        Returns
        -------
            LinearOperator: Sum of the two covariance operators.
        """
        dim = self.shape[0]
        diag_indices = jnp.diag_indices(dim)
        new_matrix = self.matrix.at[diag_indices].add(other.diagonal())

        return DenseLinearOperator(matrix=new_matrix)

    def diagonal(self) -> Float[Array, " N"]:
        """
        Diagonal of the covariance operator.

        Returns
        -------
            Float[Array, " N"]: The diagonal of the covariance operator.
        """
        return jnp.diag(self.matrix)

    def __matmul__(self, other: VecNOrMatNM) -> VecNOrMatNM:
        """Matrix multiplication.

        Args:
            other (Float[Array, "N M"]): Matrix to multiply with.

        Returns
        -------
            Float[Array, "N M"]: Result of matrix multiplication.
        """
        return jnp.matmul(self.matrix, other)

    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covariance matrix from the covariance operator.

        Returns
        -------
            Float[Array, "N N"]: Dense covariance matrix.
        """
        return self.matrix

    @classmethod
    def from_dense(cls, matrix: Float[Array, "N N"]) -> "DenseLinearOperator":
        """Construct covariance operator from dense covariance matrix.

        Args:
            matrix (Float[Array, "N N"]): Dense covariance matrix.

        Returns
        -------
            DenseLinearOperator: Covariance operator.
        """
        return DenseLinearOperator(matrix=matrix)

    @classmethod
    def from_root(cls, root: LinearOperator) -> "DenseLinearOperator":
        """Construct covariance operator from the root of the covariance matrix.

        Args:
            root (Float[Array, "N N"]): Root of the covariance matrix.

        Returns
        -------
            DenseLinearOperator: Covariance operator.
        """
        return DenseFromRootLinearOperator(root=root)


class DenseFromRootLinearOperator(DenseLinearOperator):
    root: LinearOperator

    def __init__(self, root: LinearOperator):
        """Initialize the covariance operator."""
        self.root = root
        self.shape = root.shape
        self.dtype = root.dtype

    def to_root(self) -> LinearOperator:
        return self.root

    @property
    def matrix(self) -> Float[Array, "N N"]:
        dense_root = self.root.to_dense()
        return dense_root @ dense_root.T


__all__ = [
    "DenseLinearOperator",
]
