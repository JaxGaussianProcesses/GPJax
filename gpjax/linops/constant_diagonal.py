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

from gpjax.base import static_field
from gpjax.linops.base import AbstractLinearOperator
from gpjax.linops.diagonal import Diagonal
from gpjax.linops.utils import to_linear_operator
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class ConstantDiagonal(Diagonal):
    value: Union[ScalarFloat, Float[Array, "1"]]
    size: int = static_field()
    diag: int = static_field(repr=False, init=False)

    def __init__(
        self,
        value: Union[ScalarFloat, Float[Array, "1"]],
        size: int,
        dtype: jnp.dtype = None,
    ) -> None:
        """Initialize the constant diagonal linear operator.

        Args:
            value (Union[ScalarFloat, Float[Array, "1"]]): Constant value of the diagonal.
            size (int): Size of the diagonal.
        """
        if not isinstance(size, int):
            raise ValueError(f"`length` must be an integer, but `length = {size}`.")

        value = jnp.atleast_1d(value)

        if value.ndim != 1:
            raise ValueError(
                "`value` must be one dimensional scalar, but `value.shape ="
                f" {value.shape}`."
            )

        if dtype is not None:
            value = value.astype(dtype)

        self.value = value
        self.size = size
        self.shape = (size, size)
        self.dtype = value.dtype

    def diagonal(self) -> Float[Array, " N"]:
        """Diagonal of the covariance operator."""
        return self.value * jnp.ones(self.size)

    def __matmul__(self, other: "ConstantDiagonal") -> "ConstantDiagonal":
        return ConstantDiagonal(value=self.value * other.value, size=self.size)

    def __mul__(
        self, other: Union[Number, ScalarFloat, Float[Array, "1"]]
    ) -> "ConstantDiagonal":
        return ConstantDiagonal(value=self.value * other, size=self.size)

    def __mul__(self, other: "ConstantDiagonal") -> "ConstantDiagonal":
        return ConstantDiagonal(value=self.value * other.value, size=self.size)

    def __add__(self, other: "ConstantDiagonal") -> "ConstantDiagonal":
        return ConstantDiagonal(value=self.value + other.value, size=self.size)

    def to_root(self) -> "ConstantDiagonal":
        """
        Lower triangular.

        Returns
        -------
            ConstantDiagonal: Square root of the linear operator.
        """
        return ConstantDiagonal(value=jnp.sqrt(self.value), size=self.size)

    def log_det(self) -> ScalarFloat:
        """Log determinant.

        Returns
        -------
            ScalarFloat: Log determinant of the covariance matrix.
        """
        return 2.0 * self.size * jnp.log(self.value.squeeze())

    def inverse(self) -> "ConstantDiagonal":
        """Inverse of the covariance operator.

        Returns
        -------
            Diagonal: Inverse of the covariance operator.
        """
        return ConstantDiagonal(value=1.0 / self.value, size=self.size)

    def solve(
        self, rhs: Union[AbstractLinearOperator, Float[Array, "... M"]]
    ) -> AbstractLinearOperator:
        """Solve linear system.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns
        -------
            Float[Array, "N M"]: Solution of the linear system.
        """
        return to_linear_operator(rhs / self.value)

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> "ConstantDiagonal":
        """Construct covariance operator from dense matrix.

        Args:
            dense (Float[Array, "N N"]): Dense matrix.

        Returns
        -------
            ConstantDiagonal: Linear operator.
        """
        return ConstantDiagonal(value=dense[0, 0], size=dense.shape[0])

    @classmethod
    def from_root(cls, root: "ConstantDiagonal") -> "ConstantDiagonal":
        """Construct ConstantDiagonal linear operator from root.

        Args:
            root (ConstantDiagonal): Root of the linear operator.

        Returns
        -------
            ConstantDiagonal: linear operator.
        """
        return ConstantDiagonal(value=root.value**2, size=root.size)


__all__ = [
    "ConstantDiagonal",
]
