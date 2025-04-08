"""Linear algebra operations for GPJax.

This module provides JAX-based implementations of linear algebra operations
previously provided by the cola library.
"""

from functools import singledispatch
from typing import Any, Callable, Optional, Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float


# PSD annotation replacement
class PSD:
    """A marker class to indicate that a matrix is positive semi-definite."""

    def __init__(self, matrix: Any) -> None:
        """Initialize a PSD matrix.

        Args:
            matrix: The matrix to mark as PSD.
        """
        self.matrix = matrix

    def __call__(self, *args, **kwargs):
        """Call the underlying matrix."""
        return self.matrix(*args, **kwargs)

    def to_dense(self) -> Float[Array, "N N"]:
        """Convert the matrix to a dense array."""
        if hasattr(self.matrix, "to_dense"):
            return self.matrix.to_dense()
        return self.matrix


# Linear operator base class
class LinearOperator:
    """Base class for linear operators."""

    def __init__(self, shape: Tuple[int, int], dtype=jnp.float32):
        """Initialize a linear operator.

        Args:
            shape: The shape of the operator.
            dtype: The data type of the operator.
        """
        self.shape = shape
        self.dtype = dtype
        self.annotations = set()

    def to_dense(self) -> Float[Array, "N N"]:
        """Convert the operator to a dense array."""
        raise NotImplementedError("to_dense not implemented")

    def __matmul__(self, other):
        """Matrix multiplication."""
        if isinstance(other, LinearOperator):
            return self.to_dense() @ other.to_dense()
        return self.to_dense() @ other


# Dense matrix operator
class Dense(LinearOperator):
    """Dense matrix operator."""

    def __init__(self, matrix: Float[Array, "N N"]):
        """Initialize a dense matrix operator.

        Args:
            matrix: The dense matrix.
        """
        super().__init__(matrix.shape, matrix.dtype)
        self.matrix = matrix

    def to_dense(self) -> Float[Array, "N N"]:
        """Convert the operator to a dense array."""
        return self.matrix


# Diagonal matrix operator
class Diagonal(LinearOperator):
    """Diagonal matrix operator."""

    def __init__(self, diag: Float[Array, " N"]):
        """Initialize a diagonal matrix operator.

        Args:
            diag: The diagonal elements.
        """
        super().__init__((diag.shape[0], diag.shape[0]), diag.dtype)
        self.diag = diag

    def to_dense(self) -> Float[Array, "N N"]:
        """Convert the operator to a dense array."""
        return jnp.diag(self.diag)


# Identity matrix operator
class Identity(LinearOperator):
    """Identity matrix operator."""

    def __init__(self, shape: Tuple[int, int], dtype=jnp.float32):
        """Initialize an identity matrix operator.

        Args:
            shape: The shape of the operator.
            dtype: The data type of the operator.
        """
        super().__init__(shape, dtype)

    def to_dense(self) -> Float[Array, "N N"]:
        """Convert the operator to a dense array."""
        return jnp.eye(self.shape[0], dtype=self.dtype)


# Triangular matrix operator
class Triangular(LinearOperator):
    """Triangular matrix operator."""

    def __init__(self, matrix: Float[Array, "N N"], lower: bool = True):
        """Initialize a triangular matrix operator.

        Args:
            matrix: The matrix.
            lower: Whether the matrix is lower triangular.
        """
        super().__init__(matrix.shape, matrix.dtype)
        self.matrix = matrix
        self.lower = lower

    def to_dense(self) -> Float[Array, "N N"]:
        """Convert the operator to a dense array."""
        if self.lower:
            return jnp.tril(self.matrix)
        return jnp.triu(self.matrix)


# I_like function to create an identity matrix like another matrix
def I_like(matrix: Union[LinearOperator, Float[Array, "N N"]]) -> Identity:
    """Create an identity matrix with the same shape as the input.

    Args:
        matrix: The matrix to match the shape of.

    Returns:
        An identity matrix with the same shape as the input.
    """
    if isinstance(matrix, LinearOperator):
        return Identity(matrix.shape, matrix.dtype)
    return Identity((matrix.shape[0], matrix.shape[0]), matrix.dtype)


# Solve linear system
@singledispatch
def solve(
    A: Any, b: Float[Array, "... N"], algorithm: Optional[Any] = None
) -> Float[Array, "... N"]:
    """Solve the linear system Ax = b.

    Args:
        A: The matrix A.
        b: The right-hand side b.
        algorithm: Ignored, for compatibility with cola API.

    Returns:
        The solution x.
    """
    if isinstance(A, PSD):
        return solve(A.matrix, b)

    if hasattr(A, "to_dense"):
        A_dense = A.to_dense()
    else:
        A_dense = A

    return jsp.linalg.solve(A_dense, b, assume_a="pos")


@solve.register
def _(
    A: Diagonal, b: Float[Array, "... N"], algorithm: Optional[Any] = None
) -> Float[Array, "... N"]:
    """Solve the linear system Ax = b for diagonal A."""
    return b / A.diag


@solve.register
def _(
    A: Identity, b: Float[Array, "... N"], algorithm: Optional[Any] = None
) -> Float[Array, "... N"]:
    """Solve the linear system Ax = b for identity A."""
    return b


# Matrix inverse
@singledispatch
def inv(A: Any, algorithm: Optional[Any] = None) -> Float[Array, "N N"]:
    """Compute the inverse of a matrix.

    Args:
        A: The matrix to invert.
        algorithm: Ignored, for compatibility with cola API.

    Returns:
        The inverse of A.
    """
    if isinstance(A, PSD):
        return inv(A.matrix)

    if hasattr(A, "to_dense"):
        A_dense = A.to_dense()
    else:
        A_dense = A

    return jnp.linalg.inv(A_dense)


@inv.register
def _(A: Diagonal, algorithm: Optional[Any] = None) -> Diagonal:
    """Compute the inverse of a diagonal matrix."""
    return Diagonal(1.0 / A.diag)


@inv.register
def _(A: Identity, algorithm: Optional[Any] = None) -> Identity:
    """Compute the inverse of an identity matrix."""
    return A


# Matrix determinant
@singledispatch
def logdet(
    A: Any, algorithm1: Optional[Any] = None, algorithm2: Optional[Any] = None
) -> Float[Array, ""]:
    """Compute the log determinant of a matrix.

    Args:
        A: The matrix.
        algorithm1: Ignored, for compatibility with cola API.
        algorithm2: Ignored, for compatibility with cola API.

    Returns:
        The log determinant of A.
    """
    if isinstance(A, PSD):
        return logdet(A.matrix)

    if hasattr(A, "to_dense"):
        A_dense = A.to_dense()
    else:
        A_dense = A

    # Use Cholesky decomposition for numerical stability
    L = jnp.linalg.cholesky(A_dense)
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


@logdet.register
def _(
    A: Diagonal, algorithm1: Optional[Any] = None, algorithm2: Optional[Any] = None
) -> Float[Array, ""]:
    """Compute the log determinant of a diagonal matrix."""
    return jnp.sum(jnp.log(A.diag))


@logdet.register
def _(
    A: Identity, algorithm1: Optional[Any] = None, algorithm2: Optional[Any] = None
) -> Float[Array, ""]:
    """Compute the log determinant of an identity matrix."""
    return jnp.array(0.0, dtype=A.dtype)


# Diagonal extraction
@singledispatch
def diag(A: Any) -> Float[Array, " N"]:
    """Extract the diagonal of a matrix.

    Args:
        A: The matrix.

    Returns:
        The diagonal of A.
    """
    if isinstance(A, PSD):
        return diag(A.matrix)

    if hasattr(A, "to_dense"):
        A_dense = A.to_dense()
    else:
        A_dense = A

    return jnp.diag(A_dense)


@diag.register
def _(A: Diagonal) -> Float[Array, " N"]:
    """Extract the diagonal of a diagonal matrix."""
    return A.diag


@diag.register
def _(A: Identity) -> Float[Array, " N"]:
    """Extract the diagonal of an identity matrix."""
    return jnp.ones(A.shape[0], dtype=A.dtype)


# Cholesky decomposition
class Cholesky:
    """Cholesky decomposition algorithm."""

    pass


# Auto algorithm selection
class Auto:
    """Automatic algorithm selection."""

    pass


# Conjugate gradient algorithm
class CG:
    """Conjugate gradient algorithm."""

    pass
