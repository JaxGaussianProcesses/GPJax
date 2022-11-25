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

from typing import Tuple, Union, Any

import jax.numpy as jnp
from jaxtyping import Array, Float

from .linear_operator import LinearOperator
from .dense_linear_operator import DenseLinearOperator
from .utils import to_linear_operator


def _check_diag(diag: Any) -> None:
    """Check if the diagonal is a vector."""

    if diag.ndim != 1:
        raise ValueError(
            f"The `matrix` must be a one dimension vector, but "
            f"`diag.shape = {diag.shape}`."
        )


class DiagonalLinearOperator(LinearOperator):
    """Diagonal covariance operator."""

    def __init__(self, diag: Float[Array, "N"]) -> None:
        """Initialize the covariance operator.

        Args:
            diag (Float[Array, "N"]): Diagonal of the covariance operator.
        """
        _check_diag(diag)
        self.diag = diag

    def diagonal(self) -> Float[Array, "N"]:
        """Diagonal of the covariance operator.

        Returns:
            Float[Array, "N"]: Diagonal of the covariance operator.
        """
        return self.diag

    def __add__(
        self, other: Union[LinearOperator, Float[Array, "N N"]]
    ) -> LinearOperator:
        """Add diagonal to another linear operator.

        Args:
            other (Union[LinearOperator, Float[Array, "N N"]]): Other linear operator. Dimension of both operators must match. If the other linear operator is not a DiagonalLinearOperator, dense matrix addition is used.

        Returns:
            LinearOperator: linear operator plus the diagonal linear operator.
        """

        from .zero_linear_operator import ZeroLinearOperator

        other = to_linear_operator(other)

        if isinstance(other, DiagonalLinearOperator):
            return DiagonalLinearOperator(diag=self.diagonal() + other.diagonal())

        elif isinstance(other, DenseLinearOperator):
            return other._add_diagonal(self)

        elif isinstance(other, ZeroLinearOperator):
            return self

        else:
            raise NotImplementedError

    def __mul__(self, other: float) -> LinearOperator:
        """Multiply covariance operator by scalar.

        Args:
            other (LinearOperator): Scalar.

        Returns:
            LinearOperator: Covariance operator multiplied by a scalar.
        """

        return DiagonalLinearOperator(diag=self.diagonal() * other)

    def _add_diagonal(self, other: "DiagonalLinearOperator") -> LinearOperator:
        """Add diagonal to the covariance operator,  useful for computing, Kxx + Iσ².

        Args:
            other (DiagonalLinearOperator): Diagonal covariance operator to add to the covariance operator.

        Returns:
            LinearOperator: Covariance operator with the diagonal added.
        """

        return DiagonalLinearOperator(diag=self.diagonal() + other.diagonal())

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
        return jnp.diag(self.diagonal())

    def __matmul__(self, other: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Matrix multiplication.

        Args:
            x (Float[Array, "N M"]): Matrix to multiply with.

        Returns:
            Float[Array, "N M"]: Result of matrix multiplication.
        """
        diag = (
            self.diagonal() if other.ndim == 1 else jnp.expand_dims(self.diagonal(), -1)
        )

        return diag * other

    def to_root(self) -> DiagonalLinearOperator:
        """
        Lower triangular.

        Returns:
            Float[Array, "N N"]: Lower triangular matrix.
        """
        return DiagonalLinearOperator(diag=jnp.sqrt(self.diagonal()))

    def log_det(self) -> Float[Array, "1"]:
        """Log determinant.

        Returns:
            Float[Array, "1"]: Log determinant of the covariance matrix.
        """
        return jnp.sum(jnp.log(self.diagonal()))

    def inverse(self) -> DiagonalLinearOperator:
        """Inverse of the covariance operator.

        Returns:
            DiagonalLinearOperator: Inverse of the covariance operator.
        """
        return DiagonalLinearOperator(diag=1.0 / self.diagonal())

    def solve(self, rhs: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Solve linear system.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns:
            Float[Array, "N M"]: Solution of the linear system.
        """

        return self.inverse() @ rhs

    @classmethod
    def from_root(cls, root: DiagonalLinearOperator) -> DiagonalLinearOperator:
        """Construct covariance operator from the lower triangular matrix.

        Returns:
            DiagonalLinearOperator: Covariance operator.
        """
        return _DiagonalFromRoot(root=root)

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> DiagonalLinearOperator:
        """Construct covariance operator from its dense matrix representation.

        Returns:
            DiagonalLinearOperator: Covariance operator.
        """
        return DiagonalLinearOperator(diag=dense.diagonal())


class _DiagonalFromRoot(DiagonalLinearOperator):
    def __init__(self, root: DiagonalLinearOperator):
        """Initialize the covariance operator."""

        if not isinstance(root, DiagonalLinearOperator):
            raise ValueError("root must be a DiagonalLinearOperator")

        self.root = root

    def to_root(self) -> LinearOperator:
        return self.root

    @property
    def shape(self) -> Tuple[int, int]:
        return self.root.shape

    @property
    def diag(self) -> Float[Array, "N"]:
        return self.root.diagonal() ** 2

    def diagonal(self) -> Float[Array, "N"]:
        return self.root.diagonal() ** 2


__all__ = [
    "DiagonalLinearOperator",
]
