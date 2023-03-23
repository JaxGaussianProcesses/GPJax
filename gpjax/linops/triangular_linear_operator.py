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

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float

from .linear_operator import LinearOperator
from .dense_linear_operator import DenseLinearOperator


class LowerTriangularLinearOperator(DenseLinearOperator):
    """Current implementation of the following methods is inefficient.
    We assume a dense matrix representation of the operator. But take advantage of the solve structure."""

    @property
    def T(self) -> UpperTriangularLinearOperator:
        return UpperTriangularLinearOperator(matrix=self.matrix.T)

    def to_root(self) -> LinearOperator:
        raise ValueError("Matrix is not positive semi-definite.")

    def inverse(self) -> DenseLinearOperator:
        matrix = self.solve(jnp.eye(self.size))
        return DenseLinearOperator(matrix)

    def solve(self, rhs: Float[Array, "N"]) -> Float[Array, "N"]:
        return jsp.linalg.solve_triangular(self.to_dense(), rhs, lower=True)

    def __matmul__(self, other):
        return super().__matmul__(other)

    def __add__(self, other):
        return super().__matmul__(other)

    @classmethod
    def from_root(cls, root: LinearOperator) -> None:
        raise ValueError("LowerTriangularLinearOperator does not have a root.")

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> LowerTriangularLinearOperator:
        return LowerTriangularLinearOperator(matrix=dense)


class UpperTriangularLinearOperator(DenseLinearOperator):
    """Current implementation of the following methods is inefficient.
    We assume a dense matrix representation of the operator. But take advantage of the solve structure."""

    @property
    def T(self) -> LowerTriangularLinearOperator:
        return LowerTriangularLinearOperator(matrix=self.matrix.T)

    def to_root(self) -> LinearOperator:
        raise ValueError("Matrix is not positive semi-definite.")

    def inverse(self) -> DenseLinearOperator:
        matrix = self.solve(jnp.eye(self.size))
        return DenseLinearOperator(matrix)

    def __matmul__(self, other):
        return super().__matmul__(other)

    def __add__(self, other):
        return super().__matmul__(other)

    def solve(self, rhs: Float[Array, "N"]) -> Float[Array, "N"]:
        return jsp.linalg.solve_triangular(self.to_dense(), rhs, lower=False)

    @classmethod
    def from_root(cls, root: LinearOperator) -> None:
        raise ValueError("LowerTriangularLinearOperator does not have a root.")

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> UpperTriangularLinearOperator:
        return UpperTriangularLinearOperator(matrix=dense)


__all__ = ["TriangularLinearOperator"]
