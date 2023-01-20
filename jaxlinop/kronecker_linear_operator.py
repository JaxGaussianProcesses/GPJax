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

from typing import TYPE_CHECKING, List, Union, Any

if TYPE_CHECKING:
    from .diagonal_linear_operator import DiagonalLinearOperator
    from .triangular_linear_operator import LowerTriangularLinearOperator

import jax.numpy as jnp
import operator
from functools import reduce
from jaxtyping import Array, Float

from .linear_operator import LinearOperator


def _check_linops(linops: Any) -> None:
    """Checks that the inputs are correct."""

    if not isinstance(linops, List):
        raise ValueError("The parameter `linops` must be a list.")

    for linop in linops:
        if not isinstance(linop, LinearOperator):
            raise ValueError(
                f"linops must be a list of LinearOperators, but got {type(linop)}"
            )


class KroneckerLinearOperator(LinearOperator):
    """Dense covariance operator."""

    def __init__(self, linops: List[LinearOperator], dtype: jnp.dtype = None):
        """Initialize the covariance operator.

        Args:
            linops (List[LinearOperator]): A list of linear operators.
        """
        _check_linops(linops)
        self.linops = linops

        if dtype is None:
            self._dtype = linops[0].dtype

    def shape(self) -> tuple[int, int]:
        left = reduce(operator.mul, [linop.shape[-2] for linop in self.linops], 1)
        right = reduce(operator.mul, [linop.shape[-1] for linop in self.linops], 1)
        return (left, right)

    def dtype(self) -> jnp.dtype:
        return self._dtype

    def inverse(self):
        # (A₁ ⊗ ... ⊗ Aₙ)⁻¹ = A₁⁻¹ ⊗ ... ⊗ Aₙ⁻¹
        return KroneckerLinearOperator([linop.inverse() for linop in self.linops])

    def to_root(self):
        # (A₁ ⊗ ... ⊗ Aₙ)⁻¹ = A₁⁻¹ ⊗ ... ⊗ Aₙ⁻¹
        return KroneckerLinearOperator([linop.to_root() for linop in self.linops])

    def trace(self):
        # tr(A₁ ⊗ ... ⊗ Aₙ) = tr(A₁) x ... x tr(Aₙ)
        return reduce(operator.mul, [linop.trace() for linop in self.linops], 1)

    def __add__(
        self, other: Union[LinearOperator, Float[Array, "N N"]]
    ) -> LinearOperator:
        raise NotImplementedError

    def __mul__(self, other: float) -> LinearOperator:
        raise NotImplementedError

    def _add_diagonal(self, other: DiagonalLinearOperator) -> LinearOperator:
        raise NotImplementedError

    def diagonal(self) -> Float[Array, "N"]:
        raise NotImplementedError

    def __matmul__(self, other: Float[Array, "N M"]) -> Float[Array, "N M"]:
        raise NotImplementedError

    def to_dense(self) -> Float[Array, "N N"]:
        raise NotImplementedError

    @classmethod
    def from_root(cls, root: LowerTriangularLinearOperator) -> KroneckerLinearOperator:
        """Construct covariance operator from the root of the covariance matrix.

        Args:
            root (Float[Array, "N N"]): Root of the covariance matrix.

        Returns:
            KroneckerLinearOperator: Covariance operator.
        """

        raise NotImplementedError


class _KroneckerFromRoot(KroneckerLinearOperator):
    pass


__all__ = [
    "DenseLinearOperator",
]
