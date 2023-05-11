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


from dataclasses import dataclass

from beartype.typing import (
    Any,
    Tuple,
    Union,
)
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.linops.diagonal_linear_operator import DiagonalLinearOperator
from gpjax.linops.linear_operator import LinearOperator
from gpjax.linops.utils import (
    check_shapes_match,
    default_dtype,
    to_linear_operator,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


def _check_size(shape: Any) -> None:
    if not isinstance(shape, tuple):
        raise ValueError(
            f"`shape` must be a a tuple, but `type(shape) = {type(shape)}`."
        )

    if len(shape) != 2:
        raise ValueError(f"`shape` must be a 2-tuple, but `shape = {shape}`.")


# TODO: Generalise to non-square matrices.
@dataclass
class ZeroLinearOperator(LinearOperator):
    """Zero linear operator."""

    def __init__(self, shape: Tuple[int, ...], dtype: jnp.dtype = None) -> None:
        _check_size(shape)

        if dtype is None:
            dtype = default_dtype()

        self.shape = shape
        self.dtype = dtype

    def diagonal(self) -> Float[Array, " N"]:
        """
        Diagonal of the covariance operator.

        Returns
        -------
            Float[Array, " N"]: The diagonal of the covariance operator.
        """
        return jnp.zeros(self.shape[0])

    def __add__(
        self, other: Union[Float[Array, "N N"], LinearOperator]
    ) -> Union[Float[Array, "N N"], LinearOperator]:
        """Add covariance operator to another covariance operator.

        Args:
            other (Union[Float[Array, "N N"], LinearOperator]): Covariance operator to add.

        Returns
        -------
            Union[Float[Array, "N N"], LinearOperator]: Sum of the covariance operators.
        """
        check_shapes_match(self.shape, other.shape)
        return to_linear_operator(other)

    def _add_diagonal(self, other: DiagonalLinearOperator) -> DiagonalLinearOperator:
        """Add diagonal to the covariance operator,  useful for computing, Kxx + Io².

        Args:
            other (DiagonalLinearOperator): Diagonal covariance operator to add to the covariance operator.

        Returns
        -------
            DiagonalLinearOperator: Covariance operator with the diagonal added.
        """
        check_shapes_match(self.shape, other.shape)
        return other

    def __mul__(self, other: ScalarFloat) -> "ZeroLinearOperator":
        """Multiply covariance operator by scalar.

        Args:
            other (ConstantDiagonalLinearOperator): Scalar.

        Returns
        -------
            ZeroLinearOperator: Covariance operator multiplied by a scalar.
        """
        # TODO: check shapes.
        return self

    def __matmul__(
        self, other: Union[LinearOperator, Float[Array, "N M"]]
    ) -> "ZeroLinearOperator":
        """Matrix multiplication.

        Args:
            other (Union[LinearOperator, Float[Array, "N M"]]): Matrix to multiply with.

        Returns
        -------
            Float[Array, "N M"]: Result of matrix multiplication.
        """
        check_shapes_match(self.shape, other.shape)
        return self

    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covariance matrix from the covariance operator.

        Returns
        -------
            Float[Array, "N N"]: Dense covariance matrix.
        """
        return jnp.zeros(self.shape)

    def to_root(self) -> "ZeroLinearOperator":
        """
        Root of the covariance operator.

        Returns
        -------
            ZeroLinearOperator: Root of the covariance operator.
        """
        return self

    def log_det(self) -> ScalarFloat:
        """Log determinant.

        Returns
        -------
            ScalarFloat: Log determinant of the covariance matrix.
        """
        return jnp.log(jnp.array(0.0))

    def inverse(self) -> None:
        """Inverse of the covariance operator.

        Raises
        ------
            RuntimeError: ZeroLinearOperator is not invertible.
        """
        raise RuntimeError("ZeroLinearOperator is not invertible.")

    def solve(self, rhs: Float[Array, "... M"]) -> None:
        """Solve linear system.

        Raises
        ------
            RuntimeError: ZeroLinearOperator is not invertible.
        """
        raise RuntimeError("ZeroLinearOperator is not invertible.")

    @classmethod
    def from_root(cls, root: "ZeroLinearOperator") -> "ZeroLinearOperator":
        """Construct covariance operator from the root.

        Args:
            root (ZeroLinearOperator): Root of the covariance operator.

        Returns
        -------
            ZeroLinearOperator: Covariance operator.
        """
        return root

    @classmethod
    def from_dense(cls, dense: Float[Array, "N N"]) -> "ZeroLinearOperator":
        """Construct covariance operator from the dense matrix.

        Args:
            dense (Float[Array, "N N"]): Dense matrix.

        Returns
        -------
            ZeroLinearOperator: Covariance operator.
        """
        # TODO: check shapes.
        n = dense.shape[0]
        return ZeroLinearOperator(shape=(n, n))


__all__ = [
    "ZeroLinearOperator",
]
