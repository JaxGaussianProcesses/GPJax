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
from typing import Tuple

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

    name: str = None

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
    def cholesky(self) -> Float[Array, "N N"]:
        """Compute Cholesky decomposition.

        Returns:
            Float[Array, "N N"]: Cholesky decomposition of the covariance matrix.
        """
        raise NotImplementedError

    def log_det(self) -> Float[Array, "1"]:
        """Log determinant of the covariance matrix.

        Returns:
            Float[Array, "1"]: Log determinant of the covariance matrix.
        """
        L = self.cholesky()
        return 2.0 * jsp.sum(jnp.log(jnp.diag(L)))

    def solve(self, rhs: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Solve linear system.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns:
            Float[Array, "N M"]: Solution of the linear system.
        """
        return jsp.linalg.cho_solve((self.cholesky(), True), rhs)

    def trace(self) -> Float[Array, "1"]:
        """Trace of the covariance matrix.

        Returns:
            Float[Array, "1"]: Trace of the covariance matrix.
        """
        return jnp.sum(self.diagonal())


@dataclass
class DenseCovarianceOperator(CovarianceOperator):
    """Dense covariance operator."""

    matrix: Float[Array, "N N"]
    name: str = "Dense covariance operator"

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

    def cholesky(self) -> Float[Array, "N N"]:
        """
        Cholesky decomposition.

        Returns:
            Float[Array, "N N"]: Cholesky decomposition.
        """

        return jnp.linalg.cholesky(self.matrix)


@dataclass
class DiagonalCovarianceOperator(CovarianceOperator):
    """Diagonal covariance operator."""

    diagonal: Float[Array, "N"]
    name: str = "Diagonal covariance operator"

    @property
    def shape(self) -> Tuple[int, int]:
        """Covaraince matrix shape.

        Returns:
            Tuple[int, int]: shape of the covariance operator.
        """
        N = self.diagonal.shape[0]
        return (N, N)

    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covaraince matrix from the covariance operator.

        Returns:
            Float[Array, "N N"]: Dense covariance matrix.
        """
        return jnp.diag(self.diagonal)

    def diagonal(self) -> Float[Array, "N"]:
        """
        Diagonal of the covariance operator.

        Returns:
            Float[Array, "N"]: The diagonal of the covariance operator.
        """
        return self.diagonal

    def __matmul__(self, x: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Matrix multiplication.

        Args:
            x (Float[Array, "N M"]): Matrix to multiply with.

        Returns:
            Float[Array, "N M"]: Result of matrix multiplication.
        """
        diag_mat = lax.expand_dims(self.diagonal(), -1)
        return diag_mat * x

    def cholesky(self) -> Float[Array, "N N"]:
        """
        Cholesky decomposition.

        Returns:
            Float[Array, "N N"]: Cholesky decomposition.
        """
        return jnp.diag(jnp.sqrt(self.diagonal))

    def log_det(self) -> Float[Array, "1"]:
        """Log determinant of the covariance matrix.

        Returns:
            Float[Array, "1"]: Log determinant of the covariance matrix.
        """
        return 2.0 * jnp.sum(jnp.log(self.diagonal))

    def solve(self, rhs: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Solve linear system.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns:
            Float[Array, "N M"]: Solution of the linear system.
        """
        inv_diag_mat = lax.expand_dims(1.0 / self.diagonal(), -1)
        return rhs * inv_diag_mat


__all__ = [
    "CovarianceOperator",
    "DenseCoarianceOperator",
    "DiagonalCovarianceOperator",
]
