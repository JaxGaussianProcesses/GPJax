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

from beartype.typing import Tuple
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.linops.base import AbstractLinearOperator
from gpjax.linops.utils import (
    _check_dtype_arg,
    _check_shape_arg,
    default_dtype,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class Zero(AbstractLinearOperator):
    """Zero linear operator."""

    def __init__(self, shape: Tuple[int, ...], dtype: jnp.dtype = None) -> None:
        """Initialise a zero linear operator.

        Args:
            shape (Tuple[int, ...]): Shape of the linear operator.
            dtype (jnp.dtype, optional): Data type of the linear operator. Defaults to None.
        """

        # Get dtype.
        if dtype is None:
            dtype = default_dtype()

        # Check shape.
        _check_shape_arg(shape)

        # Check dtype.
        _check_dtype_arg(dtype)

        # Set attributes.
        self.shape = shape
        self.dtype = dtype

    @property
    def T(self) -> "Zero":
        """Transpose of the covariance operator.

        Returns
        -------
            Zero: Transpose of the covariance operator.
        """
        return self.replace(shape=tuple(reversed(self.shape)))

    def __matmul__(self, other: AbstractLinearOperator) -> AbstractLinearOperator:
        return Zero((self.shape[0], other.shape[1]), dtype=self.dtype)

    def __mul__(self, other: AbstractLinearOperator) -> AbstractLinearOperator:
        return Zero(self.shape, self.dtype)

    def __add__(self, other: AbstractLinearOperator) -> AbstractLinearOperator:
        return other

    def diagonal(self) -> Float[Array, "N"]:  # noqa: F821
        """
        Diagonal of the covariance operator.

        Returns
        -------
            Float[Array, " N"]: The diagonal of the covariance operator.
        """
        return jnp.zeros(self.shape[0])

    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covariance matrix from the covariance operator.

        Returns
        -------
            Float[Array, "N N"]: Dense covariance matrix.
        """
        return jnp.zeros(self.shape, dtype=self.dtype)

    def to_root(self) -> "Zero":
        """
        Root of the covariance operator.

        Returns
        -------
            Zero: Root of the covariance operator.
        """
        return self

    def log_det(self) -> ScalarFloat:
        """Log determinant.

        Returns
        -------
            ScalarFloat: Log determinant of the covariance matrix.
        """
        return -jnp.inf

    def inverse(self) -> AbstractLinearOperator:
        """Inverse of the covariance operator.

        Raises
        ------
            RuntimeError: Zero linear operator is not invertible.
        """
        raise RuntimeError("Zero linear operator is not invertible.")

    def solve(self, rhs: Float[Array, "... M"]) -> None:
        """Solve linear system.

        Raises
        ------
            RuntimeError: Zero is not invertible.
        """
        raise RuntimeError("Zero linear operator is not invertible.")

    @classmethod
    def from_root(cls, root: "Zero") -> "Zero":
        """Construct covariance operator from the root.

        Args:
            root (Zero): Root of the covariance operator.

        Returns
        -------
            Zero: Covariance operator.
        """
        return root

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> "Zero":
        """Construct covariance operator from the dense matrix.

        Args:
            dense (Float[Array, "N N"]): Dense matrix.

        Returns
        -------
            Zero: Covariance operator.
        """
        return Zero(shape=dense.shape)


__all__ = [
    "Zero",
]
