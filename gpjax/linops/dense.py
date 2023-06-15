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


from dataclasses import dataclass
from numbers import Number

from beartype.typing import Union
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.linops.base import AbstractLinearOperator
from gpjax.linops.diagonal import Diagonal
from gpjax.typing import Array


@dataclass
class Dense(AbstractLinearOperator):
    """Dense covariance operator."""

    matrix: Union[Float[Array, "*"], Number]

    def __init__(self, matrix: Float[Array, "N M"], dtype: jnp.dtype = None) -> None:
        """Initialize the covariance operator.

        Args:
            matrix (Float[Array, "N N"]): Dense matrix.
        """
        matrix = jnp.atleast_1d(matrix)

        if dtype is not None:
            matrix = matrix.astype(dtype)

        self.matrix = matrix
        self.shape = matrix.shape
        self.dtype = matrix.dtype

    def __mul__(self, other: Diagonal) -> Diagonal:
        from gpjax.linops.diagonal import Diagonal

        return Diagonal(self.diagonal() * other.diagonal())

    def to_dense(self) -> Float[Array, "N M"]:
        """Construct dense Covariance matrix from the covariance operator.

        Returns
        -------
            Float[Array, "N M"]: Dense covariance matrix.
        """
        return self.matrix

    @classmethod
    def from_dense(cls, matrix: Float[Array, "N M"]) -> "Dense":
        """Construct covariance operator from dense covariance matrix.

        Args:
            matrix (Float[Array, "N M"]): Dense covariance matrix.

        Returns
        -------
            Dense: Covariance operator.
        """
        return Dense(matrix=matrix)

    @classmethod
    def from_root(cls, root: AbstractLinearOperator) -> "Dense":
        """Construct covariance operator from the root of the covariance matrix.

        Args:
            root (LinearOperator): Root of the covariance matrix.

        Returns
        -------
            Dense: Covariance operator.
        """
        return _DenseFromRoot(root=root)


@dataclass
class _DenseFromRoot(Dense):
    """Given the root of a square dense linear operator, construct the dense linear operator.

    This is useful for constructing a dense linear operator from a Cholesky factorization - where we don't want to recompute the Cholesky factorization.
    """

    root: AbstractLinearOperator

    def __init__(self, root: AbstractLinearOperator):
        """Initialize the covariance operator."""
        self.root = root
        self.shape = root.shape
        self.dtype = root.dtype

    def to_root(self) -> AbstractLinearOperator:
        return self.root

    @property
    def matrix(self) -> Float[Array, "N N"]:
        dense_root = self.root.to_dense()
        return dense_root @ dense_root.T


__all__ = [
    "Dense",
]
