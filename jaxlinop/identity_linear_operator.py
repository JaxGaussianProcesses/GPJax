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

from typing import Tuple, Any, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

from .constant_diagonal_linear_operator import ConstantDiagonalLinearOperator
from .utils import to_linear_operator


def _check_size(size: Any) -> None:
    """Check that size is an integer."""

    if not isinstance(size, int):
        raise ValueError(f"`size` must be an integer, but `size = {size}`.")


class IdentityLinearOperator(ConstantDiagonalLinearOperator):
    def __init__(self, size: int) -> None:
        """Identity matrix.

        Args:
            size (int): Size of the identity matrix.
        """
        _check_size(size)
        self.size = size
        self.value = jnp.array([1.0])

    def __matmul__(self, other: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Matrix multiplication.

        Args:
            other (Float[Array, "N M"]): Matrix to multiply with.

        Returns:
            Float[Array, "N M"]: Result of matrix multiplication.
        """
        return other

    def to_root(self) -> IdentityLinearOperator:
        """
        Lower triangular.

        Returns:
            Float[Array, "N N"]: Lower triangular matrix.
        """
        return self

    def log_det(self) -> Float[Array, "1"]:
        """Log determinant.

        Returns:
            Float[Array, "1"]: Log determinant of the covariance matrix.
        """
        return jnp.array(0.0)

    def inverse(self) -> ConstantDiagonalLinearOperator:
        """Inverse of the covariance operator.

        Returns:
            DiagonalLinearOperator: Inverse of the covariance operator.
        """
        return self

    def solve(self, rhs: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Solve linear system.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns:
            Float[Array, "N M"]: Solution of the linear system.
        """
        # TODO: Check shapes.

        return rhs

    @classmethod
    def from_root(cls, root: IdentityLinearOperator) -> IdentityLinearOperator:
        """Construct from root.

        Args:
            root (IdentityLinearOperator): Root of the covariance operator.

        Returns:
            IdentityLinearOperator: Covariance operator.
        """
        return root

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> IdentityLinearOperator:
        return IdentityLinearOperator(dense.shape[0])


__all__ = [
    "IdentityLinearOperator",
]
