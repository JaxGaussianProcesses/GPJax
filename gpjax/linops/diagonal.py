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

import jax.numpy as jnp
from jaxtyping import Float
from plum import ModuleType

from gpjax.linops.base import AbstractLinearOperator
from gpjax.linops.utils import to_dense
from gpjax.typing import (
    Array,
    ScalarFloat,
    VecNOrMatNM,
)

Dense = ModuleType("gpjax.linops.dense", "Dense")

ConstantDiagonal = ModuleType("gpjax.linops.constant_diagonal", "ConstantDiagonal")


@dataclass
class Diagonal(AbstractLinearOperator):
    """Diagonal covariance operator."""

    diag: Float[Array, " N"]

    def __init__(self, diag: Float[Array, " N"], dtype: jnp.dtype = None) -> None:
        """Initialize the covariance operator.

        Args:
            diag (Float[Array, " N"]): Diagonal of the covariance operator.
        """
        diag = jnp.atleast_1d(diag)

        if diag.ndim != 1:
            raise ValueError(
                "`diag` must be a one dimensional vector, but "
                f"`diag.shape = {diag.shape}`."
            )

        if dtype is not None:
            diag = diag.astype(dtype)

        dim = diag.shape[0]
        self.diag = jnp.atleast_1d(diag)
        self.shape = (dim, dim)
        self.dtype = diag.dtype

    @property
    def T(self) -> "Diagonal":
        """Transpose linear operator."""
        return self

    def diagonal(self) -> Float[Array, " N"]:
        """Diagonal of the covariance operator.

        Returns
        -------
            Float[Array, " N"]: Diagonal of the covariance operator.
        """
        return self.diag

    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covariance matrix from the covariance operator.

        Returns
        -------
            Float[Array, "N N"]: Dense covariance matrix.
        """
        return jnp.diag(self.diagonal())

    def __add__(self, other: "Diagonal") -> "Diagonal":  # noqa: F821
        return Diagonal(diag=self.diagonal().squeeze() + other.diagonal().squeeze())

    def __mul__(self, other: AbstractLinearOperator) -> "Diagonal":
        return Diagonal(diag=self.diagonal() * other.diagonal().squeeze())

    def __matmul__(self, other: Dense) -> AbstractLinearOperator:  # noqa: F821
        from gpjax.linops.dense import Dense

        diag = (
            self.diagonal() if other.ndim == 1 else jnp.expand_dims(self.diagonal(), -1)
        )

        return Dense(matrix=diag * other.to_dense())

    def __matmul__(self, other: "Diagonal") -> "Diagonal":  # noqa: F821
        return Diagonal(diag=self.diagonal() * other.diagonal())

    def __matmul__(self, other: ConstantDiagonal) -> "Diagonal":  # noqa: F821
        return Diagonal(diag=self.diagonal() * other.value)

    def to_root(self) -> "Diagonal":
        """
        Lower triangular.

        Returns
        -------
            Float[Array, "N N"]: Lower triangular matrix.
        """
        return Diagonal(jnp.sqrt(self.diagonal()))

    def log_det(self) -> ScalarFloat:
        """Log determinant.

        Returns
        -------
            ScalarFloat: Log determinant of the covariance matrix.
        """
        return jnp.sum(jnp.log(self.diagonal()))

    def inverse(self) -> "Diagonal":
        """Inverse of the covariance operator.

        Returns
        -------
            Diagonal: Inverse of the covariance operator.
        """
        return Diagonal(diag=1.0 / self.diagonal())

    def solve(self, rhs: VecNOrMatNM) -> VecNOrMatNM:
        """Solve linear system.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns
        -------
            Float[Array, "N M"]: Solution of the linear system.
        """
        return to_dense(self.inverse() @ rhs)

    @classmethod
    def from_root(cls, root: "Diagonal") -> "Diagonal":
        """Construct covariance operator from the lower triangular matrix.

        Returns
        -------
            Diagonal: Covariance operator.
        """
        return _DiagonalFromRoot(root=root)

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> "Diagonal":
        """Construct covariance operator from its dense matrix representation.

        Returns
        -------
            Diagonal: Covariance operator.
        """
        return Diagonal(diag=dense.diagonal())


class _DiagonalFromRoot(Diagonal):
    root: Diagonal

    def __init__(self, root: Diagonal):
        """Initialize the covariance operator."""
        if not isinstance(root, Diagonal):
            raise ValueError("root must be a Diagonal linear operator.")

        self.root = root
        self.shape = root.shape
        self.dtype = root.dtype

    def to_root(self) -> AbstractLinearOperator:
        return self.root

    @property
    def diag(self) -> Float[Array, " N"]:
        return self.root.diagonal() ** 2

    def diagonal(self) -> Float[Array, " N"]:
        return self.root.diagonal() ** 2


__all__ = [
    "Diagonal",
]
