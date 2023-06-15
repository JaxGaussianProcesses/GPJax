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

from beartype.typing import (
    Any,
    Union,
)
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.linops.base import AbstractLinearOperator
from gpjax.linops.constant_diagonal import ConstantDiagonal
from gpjax.linops.utils import (
    _check_dtype_arg,
    default_dtype,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


def _check_size(size: Any) -> None:
    """Check that size is an integer."""
    if not isinstance(size, int):
        raise ValueError(f"`size` must be an integer, but `size = {size}`.")


@dataclass
class Identity(ConstantDiagonal):
    """Identity linear operator."""

    def __init__(self, size: int, dtype: jnp.dtype = None) -> None:
        """Identity matrix.

        Args:
            size (int): Size of the identity matrix.
        """

        # Get dtype.
        if dtype is None:
            dtype = default_dtype()

        # Check size.
        _check_size(size)

        # Check dtype.
        _check_dtype_arg(dtype)

        # Set attributes.
        self.value = jnp.array([1.0], dtype=dtype)
        self.size = size
        self.shape = (size, size)
        self.dtype = dtype

    def __matmul__(self, other: AbstractLinearOperator) -> AbstractLinearOperator:
        return other

    def to_root(self) -> "Identity":
        """
        Lower triangular.

        Returns
        -------
            Float[Array, "N N"]: Lower triangular matrix.
        """
        return self

    def log_det(self) -> ScalarFloat:
        """Log determinant.

        Returns
        -------
            ScalarFloat: Log determinant of the covariance matrix.
        """
        return jnp.array(0.0)

    def inverse(self) -> "Identity":
        """Inverse of the covariance operator.

        Returns
        -------
            Diagonal: Inverse of the covariance operator.
        """
        return self

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
        # TODO: Check shapes.

        return rhs

    @classmethod
    def from_root(cls, root: "Identity") -> "Identity":
        """Construct from root.

        Args:
            root (Identity): Root of the covariance operator.

        Returns
        -------
            Identity: Covariance operator.
        """
        return root

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> "Identity":
        return Identity(dense.shape[0])


__all__ = [
    "Identity",
]
