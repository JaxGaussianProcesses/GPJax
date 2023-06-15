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


from beartype.typing import Union
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float

from gpjax.linops.base import AbstractLinearOperator
from gpjax.linops.dense import Dense
from gpjax.linops.utils import to_dense
from gpjax.typing import Array


class LowerTriangular(Dense):
    """Current implementation of the following methods is inefficient.
    We assume a dense matrix representation of the operator. But take advantage of the solve structure.
    """

    @property
    def T(self) -> "UpperTriangular":
        return UpperTriangular(matrix=self.matrix.T)

    def to_root(self) -> AbstractLinearOperator:
        raise ValueError("Matrix is not positive semi-definite.")

    def inverse(self) -> Dense:
        return self.solve(jnp.eye(self.size))

    def solve(self, rhs: Union[AbstractLinearOperator, Float[Array, "... M"]]) -> Dense:
        return jsp.linalg.solve_triangular(self.to_dense(), to_dense(rhs), lower=True)

    @classmethod
    def from_root(cls, root: AbstractLinearOperator) -> None:
        raise ValueError("LowerTriangular does not have a root.")

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> "LowerTriangular":
        return LowerTriangular(matrix=dense)


class UpperTriangular(Dense):
    """Current implementation of the following methods is inefficient.
    We assume a dense matrix representation of the operator. But take advantage of the solve structure.
    """

    @property
    def T(self) -> LowerTriangular:
        return LowerTriangular(matrix=self.matrix.T)

    def to_root(self) -> AbstractLinearOperator:
        raise ValueError("Matrix is not positive semi-definite.")

    def inverse(self) -> Dense:
        return self.solve(jnp.eye(self.size))

    def solve(self, rhs: Float[Array, "... M"]) -> Dense:
        return jsp.linalg.solve_triangular(self.to_dense(), to_dense(rhs), lower=False)

    @classmethod
    def from_root(cls, root: AbstractLinearOperator) -> None:
        raise ValueError("LowerTriangular does not have a root.")

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> "UpperTriangular":
        return UpperTriangular(matrix=dense)


__all__ = [
    "LowerTriangular",
    "UpperTriangular",
]
