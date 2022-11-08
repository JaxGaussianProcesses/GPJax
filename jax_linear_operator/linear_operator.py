# Copyright 2022 The Jax Linear Operator Contributors. All Rights Reserved.
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

import abc
from typing import Optional, Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
from chex import dataclass
from jaxtyping import Array, Float


@dataclass
class LinearOperator:
    """Linear operator base class."""

    name: Optional[str] = None

    def __sub__(self, other: "LinearOperator") -> "LinearOperator":
        """Subtract two linear operators.

        Args:
            other (LinearOperator): Other linear operator.

        Returns:
            LinearOperator: Difference of the two linear operators.
        """

        return self + (other * -1)

    def __rsub__(self, other: "LinearOperator") -> "LinearOperator":
        """Reimplimentation of subtracting two linear operators.

        Args:
            other (LinearOperator): Other linear operator.

        Returns:
            LinearOperator: Difference of the two linear operators.
        """
        return (self * -1) + other

    def __add__(
        self, other: Union["LinearOperator", Float[Array, "N N"]]
    ) -> "LinearOperator":
        """Add diagonal to another linear operator.

        Args:
            other (Union["LinearOperator", Float[Array, "N N"]]): Other linear operator. Dimension of both operators must match. If the other linear operator is not a DiagonalLinearOperator, dense matrix addition is used.

        Returns:
            LinearOperator: linear operator plus the diagonal linear operator.
        """

        raise NotImplementedError

    def __radd__(
        self, other: Union["LinearOperator", Float[Array, "N N"]]
    ) -> "LinearOperator":
        return self.__add__(other)

    def __mul__(self, other: float) -> "LinearOperator":
        """Multiply linear operator by scalar.

        Args:
            other (LinearOperator): Scalar.

        Returns:
            LinearOperator: linear operator multiplied by scalar.
        """

        raise NotImplementedError

    def __rmul__(self, other: float) -> "LinearOperator":
        return self.__mul__(other)

    @abc.abstractmethod
    def _add_diagonal(
        self, other: "DiagonalLinearOperator"
    ) -> "LinearOperator":
        """
        Add diagonal matrix to a linear operator, useful for computing, Kxx + Iσ².
        """
        return NotImplementedError

    @abc.abstractmethod
    def __matmul__(self, x: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Matrix multiplication.

        Args:
            x (Float[Array, "N M"]): Matrix to multiply with.

        Returns:
            Float[Array, "N M"]: Result of matrix multiplication.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Covaraince matrix shape.

        Returns:
            Tuple[int, int]: shape of the linear operator.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covaraince matrix from the linear operator.

        Returns:
            Float[Array, "N N"]: Dense linear matrix.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def diagonal(self) -> Float[Array, "N"]:
        """Construct covaraince matrix diagonal from the linear operator.

        Returns:
            Float[Array, "N"]: linear matrix diagonal.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def triangular_lower(self) -> Float[Array, "N N"]:
        """Compute lower triangular.

        Returns:
            Float[Array, "N N"]: Lower triangular of the linear matrix.
        """
        raise NotImplementedError

    def log_det(self) -> Float[Array, "1"]:
        """Log determinant of the linear matrix.

        Returns:
            Float[Array, "1"]: Log determinant of the linear matrix.
        """

        return 2.0 * jnp.sum(jnp.log(jnp.diag(self.triangular_lower())))

    def solve(self, rhs: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Solve linear system.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns:
            Float[Array, "N M"]: Solution of the linear system.
        """
        return jsp.linalg.cho_solve((self.triangular_lower(), True), rhs)

    def trace(self) -> Float[Array, "1"]:
        """Trace of the linear matrix.

        Returns:
            Float[Array, "1"]: Trace of the linear matrix.
        """
        return jnp.sum(self.diagonal())



__all__ = [
    "LinearOperator",
]
