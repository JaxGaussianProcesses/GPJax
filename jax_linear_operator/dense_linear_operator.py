# Copyright 2022 The Jax Linear Operator Contributors. All Rights Reserved.
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

from typing import Optional, Tuple, Union

import jax.numpy as jnp
from chex import dataclass
from jaxtyping import Array, Float

from .linear_operator import LinearOperator


@dataclass
class _DenseMatrix:
    matrix: Float[Array, "N N"]


@dataclass
class DenseLinearOperator(LinearOperator, _DenseMatrix):
    """Dense covariance operator."""

    name: Optional[str] = "Dense covariance operator"

    def __add__(
        self, other: Union["LinearOperator", Float[Array, "N N"]]
    ) -> "LinearOperator":
        """Add diagonal to another linear operator.

        Args:
            other (Union["LinearOperator", Float[Array, "N N"]]): Other linear operator. Dimension of both operators must match. If the other linear operator is not a DiagonalLinearOperator, dense matrix addition is used.

        Returns:
            LinearOperator: linear operator plus the diagonal linear operator.
        """

        from .diagonal_linear_operator import DiagonalLinearOperator

        if isinstance(other, DiagonalLinearOperator):
            return self._add_diagonal(other)

        elif isinstance(other, DenseLinearOperator):
            return DenseLinearOperator(matrix=self.matrix + other.matrix)

        elif isinstance(other, jnp.ndarray):
            return DenseLinearOperator(matrix= self.matrix + other)

        else:
            raise NotImplementedError

           
    def __mul__(self, other: float) -> "LinearOperator":
        """Multiply covariance operator by scalar.

        Args:
            other (LinearOperator): Scalar.

        Returns:
            LinearOperator: Covariance operator multiplied by a scalar.
        """

        return DenseLinearOperator(matrix=self.matrix * other)

    def _add_diagonal(
        self, other: "DiagonalLinearOperator"
    ) -> "LinearOperator":
        """Add diagonal to the covariance operator,  useful for computing, Kxx + Iσ².

        Args:
            other (DiagonalLinearOperator): Diagonal covariance operator to add to the covariance operator.

        Returns:
            LinearOperator: Sum of the two covariance operators.
        """

        n = self.shape[0]
        diag_indices = jnp.diag_indices(n)
        new_matrix = self.matrix.at[diag_indices].add(other.diagonal())

        return DenseLinearOperator(matrix=new_matrix)

    @property
    def shape(self) -> Tuple[int, int]:
        """Covaraince matrix shape.

        Returns:
            Tuple[int, int]: shape of the covariance operator.
        """
        return self.matrix.shape

    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covaraince matrix from the covariance operator.

        Returns:
            Float[Array, "N N"]: Dense covariance matrix.
        """
        return self.matrix

    def diagonal(self) -> Float[Array, "N"]:
        """
        Diagonal of the covariance operator.

        Returns:
            Float[Array, "N"]: The diagonal of the covariance operator.
        """

        return jnp.diag(self.matrix)

    def __matmul__(self, x: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Matrix multiplication.

        Args:
            x (Float[Array, "N M"]): Matrix to multiply with.

        Returns:
            Float[Array, "N M"]: Result of matrix multiplication.
        """

        return jnp.matmul(self.matrix, x)

    def triangular_lower(self) -> Float[Array, "N N"]:
        """Compute lower triangular.

        Returns:
            Float[Array, "N N"]: Lower triangular of the covariance matrix.
        """
        return jnp.linalg.cholesky(self.matrix)


@dataclass
class _DiagonalMatrix:
    diag: Float[Array, "N"]


@dataclass
class DiagonalLinearOperator(LinearOperator, _DiagonalMatrix):
    """Diagonal covariance operator."""

    name: Optional[str] = "Diagonal covariance operator"

    def __mul__(self, other: float) -> "LinearOperator":
        """Multiply covariance operator by scalar.

        Args:
            other (LinearOperator): Scalar.

        Returns:
            LinearOperator: Covariance operator multiplied by a scalar.
        """

        return DiagonalLinearOperator(diag=self.diag * other)

    def _add_diagonal(
        self, other: "DiagonalLinearOperator"
    ) -> "LinearOperator":
        """Add diagonal to the covariance operator,  useful for computing, Kxx + Iσ².

        Args:
            other (DiagonalLinearOperator): Diagonal covariance operator to add to the covariance operator.

        Returns:
            LinearOperator: Covariance operator with the diagonal added.
        """

        return DiagonalLinearOperator(diag=self.diag + other.diagonal())

    @property
    def shape(self) -> Tuple[int, int]:
        """Covaraince matrix shape.

        Returns:
            Tuple[int, int]: shape of the covariance operator.
        """
        N = self.diag.shape[0]
        return (N, N)

    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covaraince matrix from the covariance operator.

        Returns:
            Float[Array, "N N"]: Dense covariance matrix.
        """
        return jnp.diag(self.diag)

    def diagonal(self) -> Float[Array, "N"]:
        """
        Diagonal of the covariance operator.

        Returns:
            Float[Array, "N"]: The diagonal of the covariance operator.
        """
        return self.diag

    def __matmul__(self, x: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Matrix multiplication.

        Args:
            x (Float[Array, "N M"]): Matrix to multiply with.

        Returns:
            Float[Array, "N M"]: Result of matrix multiplication.
        """
        diag_mat = jnp.expand_dims(self.diag, -1)
        return diag_mat * x

    def triangular_lower(self) -> Float[Array, "N N"]:
        """
        Lower triangular.

        Returns:
            Float[Array, "N N"]: Lower triangular matrix.
        """
        return jnp.diag(jnp.sqrt(self.diag))

    def log_det(self) -> Float[Array, "1"]:
        """Log determinant.

        Returns:
            Float[Array, "1"]: Log determinant of the covariance matrix.
        """
        return 2.0 * jnp.sum(jnp.log(self.diag))

    def solve(self, rhs: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Solve linear system.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns:
            Float[Array, "N M"]: Solution of the linear system.
        """
        inv_diag_mat = jnp.expand_dims(1.0 / self.diag, -1)
        return rhs * inv_diag_mat


def I(n: int) -> DiagonalLinearOperator:
    """Identity matrix.

    Args:
        n (int): Size of the identity matrix.

    Returns:
        DiagonalLinearOperator: Identity matrix of shape nxn.
    """

    I = DiagonalLinearOperator(
        diag=jnp.ones(n),
        name="Identity matrix",
    )

    return I


__all__ = [
    "LinearOperator",
    "DenseCoarianceOperator",
    "DiagonalLinearOperator",
    "I",
]
