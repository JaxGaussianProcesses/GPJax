"""Utility functions for the linear algebra module."""

import jax.numpy as jnp
from jaxtyping import Array

from gpjax.linalg.operators import LinearOperator


class PSDAnnotation:
    """Marker class for PSD (Positive Semi-Definite) annotations."""

    def __call__(self, A: LinearOperator) -> LinearOperator:
        """Make PSD annotation callable."""
        return psd(A)


# Create the PSD marker similar to cola.PSD
PSD = PSDAnnotation()


def psd(A: LinearOperator) -> LinearOperator:
    """Mark a linear operator as positive semi-definite.

    This function acts as a marker/wrapper for positive semi-definite matrices.

    Args:
        A: A LinearOperator that is assumed to be positive semi-definite.

    Returns:
        The same LinearOperator, marked as PSD.
    """
    # Add annotations attribute if it doesn't exist
    if not hasattr(A, "annotations"):
        A.annotations = set()
    A.annotations.add(PSD)
    return A


def add_jitter(matrix: Array, jitter: float | Array = 1e-6) -> Array:
    """Add jitter to the diagonal of a matrix for numerical stability.

    This function adds a small positive value (jitter) to the diagonal elements
    of a square matrix to improve numerical stability, particularly for
    Cholesky decompositions and matrix inversions.

    Args:
        matrix: A square matrix to which jitter will be added.
        jitter: The jitter value to add to the diagonal. Defaults to 1e-6.

    Returns:
        The matrix with jitter added to its diagonal.

    Examples:
        >>> import jax.numpy as jnp
        >>> from gpjax.linalg.utils import add_jitter
        >>> matrix = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        >>> jittered_matrix = add_jitter(matrix, jitter=0.01)
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.ndim}D array")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")

    return matrix + jnp.eye(matrix.shape[0]) * jitter
