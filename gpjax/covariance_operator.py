# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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
from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
from chex import dataclass
from jax import lax
from jaxtyping import Array, Float


@dataclass
class CovarianceOperator:
    """Multivariate Gaussian covariance operator base class.

    Inspired by TensorFlows' LinearOperator class.
    """

    name: Optional[str] = None

    def __sub__(self, other: "CovarianceOperator") -> "CovarianceOperator":
        """Subtract two covariance operators.

        Args:
            other (CovarianceOperator): Other covariance operator.

        Returns:
            CovarianceOperator: Difference of the two covariance operators.
        """

        return self + (other * -1)

    def __rsub__(self, other: "CovarianceOperator") -> "CovarianceOperator":
        """Reimplimentation of subtracting two covariance operators.

        Args:
            other (CovarianceOperator): Other covariance operator.

        Returns:
            CovarianceOperator: Difference of the two covariance operators.
        """
        return (self * -1) + other

    def __add__(
        self, other: Union["CovarianceOperator", Float[Array, "N N"]]
    ) -> "CovarianceOperator":
        """Add diagonal to another covariance operator.

        Args:
            other (Union["CovarianceOperator", Float[Array, "N N"]]): Other
                covariance operator. Dimension of both operators must match.
                If the other covariance operator is not a
                DiagonalCovarianceOperator, dense matrix addition is used.

        Returns:
            CovarianceOperator: Covariance operator plus the diagonal covariance operator.
        """

        # Check shapes:
        if not (other.shape == self.shape):
            raise ValueError(
                f"Shape mismatch: {self.shape} and {other.shape} are not equal."
            )

        # If other is a JAX array, we convert it to a DenseCovarianceOperator
        if isinstance(other, jnp.ndarray):
            other = DenseCovarianceOperator(matrix=other)

        # Matix addition:
        if isinstance(other, DiagonalCovarianceOperator):
            return self._add_diagonal(other)

        if isinstance(self, DiagonalCovarianceOperator):
            return other._add_diagonal(self)

        elif isinstance(other, CovarianceOperator):

            return DenseCovarianceOperator(matrix=self.to_dense() + other.to_dense())

        else:
            raise NotImplementedError

    def __radd__(
        self, other: Union["CovarianceOperator", Float[Array, "N N"]]
    ) -> "CovarianceOperator":
        return self.__add__(other)

    def __mul__(self, other: float) -> "CovarianceOperator":
        """Multiply covariance operator by scalar.

        Args:
            other (CovarianceOperator): Scalar.

        Returns:
            CovarianceOperator: Covariance operator multiplied by scalar.
        """

        raise NotImplementedError

    def __rmul__(self, other: float) -> "CovarianceOperator":
        return self.__mul__(other)

    @abc.abstractmethod
    def _add_diagonal(
        self, other: "DiagonalCovarianceOperator"
    ) -> "CovarianceOperator":
        """
        Add diagonal matrix to a linear operator, useful for computing, Kxx + Iσ².
        """
        return NotImplementedError

    @abc.abstractmethod
    def __matmul__(self, x: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Matrix multiplication.

        Args:
            x (Float[Array, "N M"]): Matrix to multiply with.

        Returns:
            Float[Array, "N M"]: Result of matrix multiplication.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Covaraince matrix shape.

        Returns:
            Tuple[int, int]: shape of the covariance operator.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covaraince matrix from the covariance operator.

        Returns:
            Float[Array, "N N"]: Dense covariance matrix.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def diagonal(self) -> Float[Array, "N"]:
        """Construct covaraince matrix diagonal from the covariance operator.

        Returns:
            Float[Array, "N"]: Covariance matrix diagonal.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def triangular_lower(self) -> Float[Array, "N N"]:
        """Compute lower triangular.

        Returns:
            Float[Array, "N N"]: Lower triangular of the covariance matrix.
        """
        raise NotImplementedError

    def log_det(self) -> Float[Array, "1"]:
        """Log determinant of the covariance matrix.

        Returns:
            Float[Array, "1"]: Log determinant of the covariance matrix.
        """

        return 2.0 * jnp.sum(jnp.log(jnp.diag(self.triangular_lower())))

    def solve(self, rhs: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Solve linear system.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns:
            Float[Array, "N M"]: Solution of the linear system.
        """
        return jsp.linalg.cho_solve((self.triangular_lower(), True), rhs)

    def trace(self) -> Float[Array, "1"]:
        """Trace of the covariance matrix.

        Returns:
            Float[Array, "1"]: Trace of the covariance matrix.
        """
        return jnp.sum(self.diagonal())


@dataclass
class _DenseMatrix:
    matrix: Float[Array, "N N"]


@dataclass
class DenseCovarianceOperator(CovarianceOperator, _DenseMatrix):
    """Dense covariance operator."""

    name: Optional[str] = "Dense covariance operator"

    def __mul__(self, other: float) -> "CovarianceOperator":
        """Multiply covariance operator by scalar.

        Args:
            other (CovarianceOperator): Scalar.

        Returns:
            CovarianceOperator: Covariance operator multiplied by a scalar.
        """

        return DenseCovarianceOperator(matrix=self.matrix * other)

    def _add_diagonal(
        self, other: "DiagonalCovarianceOperator"
    ) -> "CovarianceOperator":
        """Add diagonal to the covariance operator,  useful for
        computing, :math:`\\mathbf{K}_{xx} + \\mathbf{I}\\sigma^2`.

        Args:
            other (DiagonalCovarianceOperator): Diagonal covariance
            operator to add to the covariance operator.

        Returns:
            CovarianceOperator: Sum of the two covariance operators.
        """

        n = self.shape[0]
        diag_indices = jnp.diag_indices(n)
        new_matrix = self.matrix.at[diag_indices].add(other.diagonal())

        return DenseCovarianceOperator(matrix=new_matrix)

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
class DiagonalCovarianceOperator(CovarianceOperator, _DiagonalMatrix):
    """Diagonal covariance operator."""

    name: Optional[str] = "Diagonal covariance operator"

    def __mul__(self, other: float) -> "CovarianceOperator":
        """Multiply covariance operator by scalar.

        Args:
            other (CovarianceOperator): Scalar.

        Returns:
            CovarianceOperator: Covariance operator multiplied by a scalar.
        """

        return DiagonalCovarianceOperator(diag=self.diag * other)

    def _add_diagonal(
        self, other: "DiagonalCovarianceOperator"
    ) -> "CovarianceOperator":
        """Add diagonal to the covariance operator,  useful for computing,
        :math:`\\mathbf{K}_{xx} + \\mathbf{I}\\sigma^2`

        Args:
            other (DiagonalCovarianceOperator): Diagonal covariance
            operator to add to the covariance operator.

        Returns:
            CovarianceOperator: Covariance operator with the diagonal added.
        """

        return DiagonalCovarianceOperator(diag=self.diag + other.diagonal())

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


def I(n: int) -> DiagonalCovarianceOperator:
    """Identity matrix.

    Args:
        n (int): Size of the identity matrix.

    Returns:
        DiagonalCovarianceOperator: Identity matrix of shape nxn.
    """

    I = DiagonalCovarianceOperator(
        diag=jnp.ones(n),
        name="Identity matrix",
    )

    return I


__all__ = [
    "CovarianceOperator",
    "DenseCoarianceOperator",
    "DiagonalCovarianceOperator",
    "I",
]
