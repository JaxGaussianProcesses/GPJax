"""Compatibility layer for linear algebra operations.

This module provides JAX-based implementations of linear algebra operations
that match the API of the cola library. This allows for easy switching between
the two implementations by only changing imports.

To use cola:
    from cola.annotations import PSD
    from cola.linalg.decompositions.decompositions import Cholesky
    from cola.linalg.inverse.inv import solve
    from cola.ops.operators import Dense, I_like, Triangular

To use JAX-based implementation:
    from gpjax.compat import PSD, Cholesky, solve, Dense, I_like, Triangular
"""

from __future__ import annotations
from functools import singledispatch
from typing import Any, Callable, Optional, Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float

# -----------------------------------------------------------------------------
# Linear Operator Classes
# -----------------------------------------------------------------------------


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

    def __mul__(self, other):
        """Scalar multiplication."""
        if isinstance(other, (int, float)):
            # For scalar multiplication, multiply the matrix elements
            return Dense(self.matrix * other)
        # For other types, convert to dense and multiply
        if hasattr(other, "to_dense"):
            return Dense(self.matrix * other.to_dense())
        return Dense(self.matrix * other)

    def __rmul__(self, other):
        """Reverse scalar multiplication."""
        if isinstance(other, (int, float)):
            return Dense(other * self.matrix)
        if hasattr(other, "to_dense"):
            return Dense(other.to_dense() * self.matrix)
        return Dense(other * self.matrix)

    def __add__(self, other):
        """Addition."""
        if hasattr(other, "to_dense"):
            # Adding to another linear operator
            return Dense(self.matrix + other.to_dense())
        # Adding to a dense array
        return Dense(self.matrix + other)

    def __radd__(self, other):
        """Reverse addition."""
        if hasattr(other, "to_dense"):
            return Dense(other.to_dense() + self.matrix)
        return Dense(other + self.matrix)

    def __rmatmul__(self, other):
        """Reverse matrix multiplication."""
        if hasattr(other, "to_dense"):
            # Matrix multiplication with another linear operator
            return Dense(other.to_dense() @ self.matrix)
        # Matrix multiplication with a dense array
        return other @ self.matrix

    def __sub__(self, other):
        """Subtraction."""
        if hasattr(other, "to_dense"):
            # Subtraction with another linear operator
            return Dense(self.matrix - other.to_dense())
        # Subtraction with a dense array
        return Dense(self.matrix - other)

    def __rsub__(self, other):
        """Reverse subtraction."""
        if hasattr(other, "to_dense"):
            # Reverse subtraction with another linear operator
            return Dense(other.to_dense() - self.matrix)
        # Reverse subtraction with a dense array
        return Dense(other - self.matrix)

    @property
    def T(self) -> Dense:
        return Dense(jnp.transpose(self.to_dense()))


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

    def __mul__(self, other):
        """Scalar multiplication."""
        if isinstance(other, (int, float)):
            # For scalar multiplication, multiply the diagonal elements
            return Diagonal(self.diag * other)
        # For other types, convert to dense and multiply
        return Dense(self.to_dense() * other)

    def __rmul__(self, other):
        """Reverse scalar multiplication."""
        return self.__mul__(other)

    def __add__(self, other):
        """Addition."""
        if isinstance(other, Diagonal):
            # Adding two diagonal matrices
            return Diagonal(self.diag + other.diag)
        elif isinstance(other, Identity):
            # Adding an identity matrix
            return Diagonal(self.diag + jnp.ones_like(self.diag))
        elif hasattr(other, "to_dense"):
            # Adding to another linear operator
            return Dense(self.to_dense() + other.to_dense())
        else:
            # Adding to a dense array
            return self.to_dense() + other

    def __radd__(self, other):
        """Reverse addition."""
        return self.__add__(other)

    def __matmul__(self, other):
        """Matrix multiplication."""
        if isinstance(other, Diagonal):
            # Matrix multiplication with another diagonal matrix
            return Diagonal(self.diag * other.diag)
        elif hasattr(other, "to_dense"):
            # Matrix multiplication with another linear operator
            return Dense(self.to_dense() @ other.to_dense())
        else:
            # Matrix multiplication with a dense array
            return self.to_dense() @ other

    def __rmatmul__(self, other):
        """Reverse matrix multiplication."""
        if isinstance(other, Diagonal):
            # Matrix multiplication with another diagonal matrix
            return Diagonal(other.diag * self.diag)
        elif hasattr(other, "to_dense"):
            # Matrix multiplication with another linear operator
            return Dense(other.to_dense() @ self.to_dense())
        else:
            # Matrix multiplication with a dense array
            return other @ self.to_dense()

    def __sub__(self, other):
        """Subtraction."""
        if isinstance(other, Diagonal):
            # Subtraction with another diagonal matrix
            return Diagonal(self.diag - other.diag)
        elif hasattr(other, "to_dense"):
            # Subtraction with another linear operator
            return Dense(self.to_dense() - other.to_dense())
        else:
            # Subtraction with a dense array
            return Dense(self.to_dense() - other)

    def __rsub__(self, other):
        """Reverse subtraction."""
        if isinstance(other, Diagonal):
            # Reverse subtraction with another diagonal matrix
            return Diagonal(other.diag - self.diag)
        elif hasattr(other, "to_dense"):
            # Reverse subtraction with another linear operator
            return Dense(other.to_dense() - self.to_dense())
        else:
            # Reverse subtraction with a dense array
            return Dense(other - self.to_dense())

    @property
    def T(self) -> Diagonal:
        return self


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

    def __mul__(self, other):
        """Scalar multiplication."""
        if isinstance(other, (int, float)):
            # For scalar multiplication, return a diagonal matrix with the scalar on the diagonal
            return Diagonal(jnp.ones(self.shape[0], dtype=self.dtype) * other)
        # For other types, convert to dense and multiply
        return Dense(self.to_dense() * other)

    def __rmul__(self, other):
        """Reverse scalar multiplication."""
        return self.__mul__(other)

    def __add__(self, other):
        """Addition."""
        if isinstance(other, Identity):
            # Adding two identity matrices
            return Diagonal(jnp.ones(self.shape[0], dtype=self.dtype) * 2)
        elif hasattr(other, "to_dense"):
            # Adding to another linear operator
            return Dense(self.to_dense() + other.to_dense())
        else:
            # Adding to a dense array
            return self.to_dense() + other

    def __radd__(self, other):
        """Reverse addition."""
        return self.__add__(other)

    def __matmul__(self, other):
        """Matrix multiplication."""
        if hasattr(other, "to_dense"):
            # Matrix multiplication with another linear operator
            return other.to_dense()
        # Matrix multiplication with a dense array
        return other

    def __rmatmul__(self, other):
        """Reverse matrix multiplication."""
        if hasattr(other, "to_dense"):
            # Matrix multiplication with another linear operator
            return other.to_dense()
        # Matrix multiplication with a dense array
        return other

    def __sub__(self, other):
        """Subtraction."""
        if isinstance(other, Identity):
            # Subtracting identity from identity gives zero
            return Dense(jnp.zeros((self.shape[0], self.shape[0]), dtype=self.dtype))
        elif hasattr(other, "to_dense"):
            # Subtraction with another linear operator
            return Dense(self.to_dense() - other.to_dense())
        else:
            # Subtraction with a dense array
            return Dense(self.to_dense() - other)

    def __rsub__(self, other):
        """Reverse subtraction."""
        if isinstance(other, Identity):
            # Subtracting identity from identity gives zero
            return Dense(jnp.zeros((self.shape[0], self.shape[0]), dtype=self.dtype))
        elif hasattr(other, "to_dense"):
            # Reverse subtraction with another linear operator
            return Dense(other.to_dense() - self.to_dense())
        else:
            # Reverse subtraction with a dense array
            return Dense(other - self.to_dense())

    @property
    def T(self) -> Identity:
        return self


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

    def __sub__(self, other):
        """Subtraction."""
        if isinstance(other, Triangular) and self.lower == other.lower:
            # Subtraction with another triangular matrix of the same type
            return Triangular(self.matrix - other.matrix, lower=self.lower)
        elif hasattr(other, "to_dense"):
            # Subtraction with another linear operator
            return Dense(self.to_dense() - other.to_dense())
        else:
            # Subtraction with a dense array
            return Dense(self.to_dense() - other)

    def __rsub__(self, other):
        """Reverse subtraction."""
        if isinstance(other, Triangular) and self.lower == other.lower:
            # Reverse subtraction with another triangular matrix of the same type
            return Triangular(other.matrix - self.matrix, lower=self.lower)
        elif hasattr(other, "to_dense"):
            # Reverse subtraction with another linear operator
            return Dense(other.to_dense() - self.to_dense())
        else:
            # Reverse subtraction with a dense array
            return Dense(other - self.to_dense())

    def __matmul__(self, other):
        """Matrix multiplication."""
        if hasattr(other, "to_dense"):
            # Matrix multiplication with another linear operator
            return Dense(self.to_dense() @ other.to_dense())
        # Matrix multiplication with a dense array
        return self.to_dense() @ other

    def __rmatmul__(self, other):
        """Reverse matrix multiplication."""
        if hasattr(other, "to_dense"):
            # Matrix multiplication with another linear operator
            return Dense(other.to_dense() @ self.to_dense())
        # Matrix multiplication with a dense array
        return other @ self.to_dense()

    def __add__(self, other):
        """Addition."""
        if isinstance(other, Triangular) and self.lower == other.lower:
            # Addition with another triangular matrix of the same type
            return Triangular(self.matrix + other.matrix, lower=self.lower)
        elif hasattr(other, "to_dense"):
            # Addition with another linear operator
            return Dense(self.to_dense() + other.to_dense())
        else:
            # Addition with a dense array
            return Dense(self.to_dense() + other)

    def __radd__(self, other):
        """Reverse addition."""
        if isinstance(other, Triangular) and self.lower == other.lower:
            # Reverse addition with another triangular matrix of the same type
            return Triangular(other.matrix + self.matrix, lower=self.lower)
        elif hasattr(other, "to_dense"):
            # Reverse addition with another linear operator
            return Dense(other.to_dense() + self.to_dense())
        else:
            # Reverse addition with a dense array
            return Dense(other + self.to_dense())

    def __mul__(self, other):
        """Scalar multiplication."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            return Triangular(self.matrix * other, lower=self.lower)
        elif hasattr(other, "to_dense"):
            # Element-wise multiplication with another linear operator
            return Dense(self.to_dense() * other.to_dense())
        else:
            # Element-wise multiplication with a dense array
            return Dense(self.to_dense() * other)

    def __rmul__(self, other):
        """Reverse scalar multiplication."""
        if isinstance(other, (int, float)):
            # Reverse scalar multiplication
            return Triangular(other * self.matrix, lower=self.lower)
        elif hasattr(other, "to_dense"):
            # Reverse element-wise multiplication with another linear operator
            return Dense(other.to_dense() * self.to_dense())
        else:
            # Reverse element-wise multiplication with a dense array
            return Dense(other * self.to_dense())

    @property
    def T(self) -> Triangular:
        return self


class BlockDiag(LinearOperator):
    """Block diagonal matrix operator."""

    def __init__(self, *Ms, multiplicities=None):
        """Initialize a block diagonal matrix operator.

        Args:
            *Ms: The matrices to put on the diagonal.
            multiplicities: The number of times to repeat each matrix.
        """
        self.Ms = Ms
        self.multiplicities = multiplicities or [1] * len(Ms)

        # Calculate shape
        total_rows = sum(M.shape[0] * mult for M, mult in zip(Ms, self.multiplicities))
        total_cols = sum(M.shape[1] * mult for M, mult in zip(Ms, self.multiplicities))

        super().__init__((total_rows, total_cols), Ms[0].dtype)

    def to_dense(self) -> Float[Array, "N N"]:
        """Convert the operator to a dense array."""
        blocks = []
        for M, mult in zip(self.Ms, self.multiplicities):
            M_dense = M.to_dense() if hasattr(M, "to_dense") else M
            blocks.extend([M_dense] * mult)

        return jsp.linalg.block_diag(*blocks)

    @property
    def T(self) -> BlockDiag:
        return BlockDiag(jnp.transpose(self.to_dense()))


class Kronecker(LinearOperator):
    """Kronecker product matrix operator."""

    def __init__(self, *Ms):
        """Initialize a Kronecker product matrix operator.

        Args:
            *Ms: The matrices to take the Kronecker product of.
        """
        self.Ms = Ms

        # Calculate shape
        rows = 1
        cols = 1
        for M in Ms:
            rows *= M.shape[0]
            cols *= M.shape[1]

        super().__init__((rows, cols), Ms[0].dtype)

    def to_dense(self) -> Float[Array, "N N"]:
        """Convert the operator to a dense array."""
        result = (
            self.Ms[0].to_dense() if hasattr(self.Ms[0], "to_dense") else self.Ms[0]
        )

        for M in self.Ms[1:]:
            M_dense = M.to_dense() if hasattr(M, "to_dense") else M
            result = jnp.kron(result, M_dense)

        return result

    @property
    def T(self) -> Kronecker:
        return Kronecker(jnp.transpose(self.to_dense()))


# -----------------------------------------------------------------------------
# Annotations
# -----------------------------------------------------------------------------


class PSD(LinearOperator):
    """A marker class to indicate that a matrix is positive semi-definite."""

    def __init__(self, matrix: Any) -> None:
        """Initialize a PSD matrix.

        Args:
            matrix: The matrix to mark as PSD.
        """
        # If the matrix is already a PSD object, just use its underlying matrix
        if isinstance(matrix, PSD):
            self.matrix = matrix.matrix
        else:
            self.matrix = matrix

        if hasattr(self.matrix, "annotations"):
            self.matrix.annotations.add(PSD)

        # Initialize LinearOperator with the shape and dtype of the matrix
        if hasattr(self.matrix, "shape") and hasattr(self.matrix, "dtype"):
            super().__init__(self.matrix.shape, self.matrix.dtype)
        elif hasattr(self.matrix, "shape"):
            super().__init__(self.matrix.shape, jnp.float32)
        else:
            # Default shape and dtype if not available
            super().__init__((1, 1), jnp.float32)

        # Add PSD to our own annotations
        self.annotations.add(PSD)

    def __call__(self, *args, **kwargs):
        """Call the underlying matrix."""
        return self.matrix(*args, **kwargs)

    def to_dense(self) -> Float[Array, "N N"]:
        """Convert the matrix to a dense array."""
        if hasattr(self.matrix, "to_dense"):
            return self.matrix.to_dense()
        return self.matrix

    def __getattr__(self, name):
        """Delegate attribute access to the underlying matrix."""
        return getattr(self.matrix, name)

    def __add__(self, other):
        """Add operation."""
        if isinstance(other, PSD):
            return PSD(self.matrix + other.matrix)
        return PSD(self.matrix + other)

    def __radd__(self, other):
        """Reverse add operation."""
        return self.__add__(other)

    def __matmul__(self, other):
        """Matrix multiplication."""
        if isinstance(other, PSD):
            return self.matrix @ other.matrix
        return self.matrix @ other

    def __rmatmul__(self, other):
        """Reverse matrix multiplication."""
        return other @ self.matrix

    def __mul__(self, other):
        """Scalar multiplication."""
        return PSD(self.matrix * other)

    def __rmul__(self, other):
        """Reverse scalar multiplication."""
        return PSD(other * self.matrix)

    def __sub__(self, other):
        """Subtraction."""
        if isinstance(other, PSD):
            return PSD(self.matrix - other.matrix)
        return PSD(self.matrix - other)

    def __rsub__(self, other):
        """Reverse subtraction."""
        if isinstance(other, PSD):
            return PSD(other.matrix - self.matrix)
        return PSD(other - self.matrix)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def I_like(matrix: Union[LinearOperator, PSD, Float[Array, "N N"]]) -> Identity:
    """Create an identity matrix with the same shape as the input.

    Args:
        matrix: The matrix to match the shape of.

    Returns:
        An identity matrix with the same shape as the input.
    """
    if isinstance(matrix, PSD):
        return I_like(matrix.matrix)
    if isinstance(matrix, LinearOperator):
        return Identity(matrix.shape, matrix.dtype)
    return Identity((matrix.shape[0], matrix.shape[0]), matrix.dtype)


# -----------------------------------------------------------------------------
# Linear Algebra Operations
# -----------------------------------------------------------------------------


# Algorithm classes for compatibility with cola API
class Algorithm:
    """Base class for linear algebra algorithms."""

    pass


class Cholesky(Algorithm):
    """Cholesky decomposition algorithm."""

    pass


class Auto(Algorithm):
    """Automatic algorithm selection."""

    pass


class CG(Algorithm):
    """Conjugate gradient algorithm."""

    pass


# Solve linear system
@singledispatch
def solve(
    A: Any, b: Float[Array, "... N"], algorithm: Optional[Algorithm] = None
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
        return solve(A.matrix, b, algorithm)

    if hasattr(A, "to_dense"):
        A_dense = A.to_dense()
    else:
        A_dense = A

    return jsp.linalg.solve(A_dense, b, assume_a="pos")


@solve.register
def _(
    A: Diagonal, b: Float[Array, "... N"], algorithm: Optional[Algorithm] = None
) -> Float[Array, "... N"]:
    """Solve the linear system Ax = b for diagonal A."""
    return b / A.diag


@solve.register
def _(
    A: Identity, b: Float[Array, "... N"], algorithm: Optional[Algorithm] = None
) -> Float[Array, "... N"]:
    """Solve the linear system Ax = b for identity A."""
    return b


# Matrix inverse
@singledispatch
def inv(A: Any, algorithm: Optional[Algorithm] = None) -> Float[Array, "N N"]:
    """Compute the inverse of a matrix.

    Args:
        A: The matrix to invert.
        algorithm: The algorithm to use for the inversion.

    Returns:
        The inverse of A.
    """
    if isinstance(A, PSD):
        return inv(A.matrix, algorithm)

    if hasattr(A, "to_dense"):
        A_dense = A.to_dense()
    else:
        A_dense = A

    return jnp.linalg.inv(A_dense)


@inv.register
def _(A: Diagonal, algorithm: Optional[Algorithm] = None) -> Diagonal:
    """Compute the inverse of a diagonal matrix."""
    return Diagonal(1.0 / A.diag)


@inv.register
def _(A: Identity, algorithm: Optional[Algorithm] = None) -> Identity:
    """Compute the inverse of an identity matrix."""
    return A


# Matrix determinant
@singledispatch
def logdet(
    A: Any,
    algorithm1: Optional[Algorithm] = None,
    algorithm2: Optional[Algorithm] = None,
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
        return logdet(A.matrix, algorithm1, algorithm2)

    if hasattr(A, "to_dense"):
        A_dense = A.to_dense()
    else:
        A_dense = A

    # Use Cholesky decomposition for numerical stability
    L = jnp.linalg.cholesky(A_dense)
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


@logdet.register
def _(
    A: Diagonal,
    algorithm1: Optional[Algorithm] = None,
    algorithm2: Optional[Algorithm] = None,
) -> Float[Array, ""]:
    """Compute the log determinant of a diagonal matrix."""
    return jnp.sum(jnp.log(A.diag))


@logdet.register
def _(
    A: Identity,
    algorithm1: Optional[Algorithm] = None,
    algorithm2: Optional[Algorithm] = None,
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


# Function dispatch mechanism
def dispatch(func):
    """Decorator to create a single-dispatch generic function."""
    return singledispatch(func)
