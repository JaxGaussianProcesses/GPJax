"""Triangular linear operator Module."""
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


import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float

from gpjax.linops.dense_linear_operator import DenseLinearOperator
from gpjax.linops.linear_operator import LinearOperator
from gpjax.typing import Array


class LowerTriangularLinearOperator(DenseLinearOperator):
    r"""Lower triangular linear operator.

    Current implementation of the following methods is inefficient.
    We assume a dense matrix representation of the operator. But take advantage of the
    solve structure.
    """

    @property
    def T(self) -> "UpperTriangularLinearOperator":
        r"""Transpose of the operator."""
        return UpperTriangularLinearOperator(matrix=self.matrix.T)

    def to_root(self) -> LinearOperator:
        r"""Square root of the operator."""
        raise ValueError("Matrix is not positive semi-definite.")

    def inverse(self) -> DenseLinearOperator:
        r"""Inverse of the operator."""
        matrix = self.solve(jnp.eye(self.size))
        return DenseLinearOperator(matrix)

    def solve(self, rhs: Float[Array, "... M"]) -> Float[Array, "... M"]:
        r"""Solve the linear system.

        Args:
            rhs (Float[Array, '... M']): Right hand side of the linear system.

        Returns:
            Float[Array, '... M']: The solution to the linear system.
        """
        return jsp.linalg.solve_triangular(self.to_dense(), rhs, lower=True)

    @classmethod
    def from_root(cls, root: LinearOperator) -> None:
        r"""Construct a lower triangular linear operator from a root. This is not
        possible for a `LowerTriangularLinearOperator` linear operator.
        """
        raise ValueError("LowerTriangularLinearOperator does not have a root.")

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> "LowerTriangularLinearOperator":
        r"""Construct a lower triangular linear operator from a dense matrix."""
        return LowerTriangularLinearOperator(matrix=dense)


class UpperTriangularLinearOperator(DenseLinearOperator):
    """Current implementation of the following methods is inefficient.
    We assume a dense matrix representation of the operator. But take advantage of the solve structure.
    """

    @property
    def T(self) -> LowerTriangularLinearOperator:
        """Transpose of the operator."""
        return LowerTriangularLinearOperator(matrix=self.matrix.T)

    def to_root(self) -> LinearOperator:
        r"""Square root of the operator."""
        raise ValueError("Matrix is not positive semi-definite.")

    def inverse(self) -> DenseLinearOperator:
        r"""Inverse of the operator."""
        matrix = self.solve(jnp.eye(self.size))
        return DenseLinearOperator(matrix)

    def solve(self, rhs: Float[Array, "... M"]) -> Float[Array, "... M"]:
        r"""Solve the linear system.

        Args:
            rhs (Float[Array, '... M']): Right hand side of the linear system.

        Returns:
            Float[Array, '... M']: The solution to the linear system.
        """
        return jsp.linalg.solve_triangular(self.to_dense(), rhs, lower=False)

    @classmethod
    def from_root(cls, root: LinearOperator) -> None:
        r"""Construct a lower triangular linear operator from a root. This is not
        possible for a `UpperTriangularLinearOperator` linear operator.
        """
        raise ValueError("UpperTriangularLinearOperator does not have a root.")

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> "UpperTriangularLinearOperator":
        r"""Construct an upper triangular linear operator from a dense matrix."""
        return UpperTriangularLinearOperator(matrix=dense)


__all__ = ["LowerTriangularLinearOperator", "UpperTriangularLinearOperator"]
