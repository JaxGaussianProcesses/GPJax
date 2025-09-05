"""Utility functions for the linear algebra module."""

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
