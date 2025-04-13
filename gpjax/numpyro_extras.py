import math

import jax
import jax.numpy as jnp
from numpyro.distributions.transforms import Transform

# -----------------------------------------------------------------------------
# Implementation: FillTriangularTransform
# -----------------------------------------------------------------------------


class FillTriangularTransform(Transform):
    """
    Transform that maps a vector of length n(n+1)/2 to an n x n lower triangular matrix.
    The ordering is assumed to be:
       (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ..., (n-1, n-1)
    """

    # Note: The base class provides `inv` through _InverseTransform wrapping _inverse.

    def __call__(self, x):
        """
        Forward transformation.

        Parameters
        ----------
        x : array_like, shape (..., L)
            Input vector with L = n(n+1)/2 for some integer n.

        Returns
        -------
        y : array_like, shape (..., n, n)
            Lower-triangular matrix (with zeros in the upper triangle) filled in
            row-major order (i.e. [ (0,0), (1,0), (1,1), ... ]).
        """
        L = x.shape[-1]
        # Use static (Python) math.sqrt to compute n. This avoids tracer issues.
        n = int((-1 + math.sqrt(1 + 8 * L)) // 2)
        if n * (n + 1) // 2 != L:
            raise ValueError("Last dimension must equal n(n+1)/2 for some integer n.")

        def fill_single(vec):
            out = jnp.zeros((n, n), dtype=vec.dtype)
            row, col = jnp.tril_indices(n)
            return out.at[row, col].set(vec)

        if x.ndim == 1:
            return fill_single(x)
        else:
            batch_shape = x.shape[:-1]
            flat_x = x.reshape((-1, L))
            out = jax.vmap(fill_single)(flat_x)
            return out.reshape(batch_shape + (n, n))

    def _inverse(self, y):
        """
        Inverse transformation.

        Parameters
        ----------
        y : array_like, shape (..., n, n)
            Lower triangular matrix.

        Returns
        -------
        x : array_like, shape (..., n(n+1)/2)
            The vector containing the elements from the lower-triangular portion of y.
        """
        if y.ndim < 2:
            raise ValueError("Input to inverse must be at least two-dimensional.")
        n = y.shape[-1]
        if y.shape[-2] != n:
            raise ValueError(
                "Input matrix must be square; got shape %s" % str(y.shape[-2:])
            )

        row, col = jnp.tril_indices(n)

        def inv_single(mat):
            return mat[row, col]

        if y.ndim == 2:
            return inv_single(y)
        else:
            batch_shape = y.shape[:-2]
            flat_y = y.reshape((-1, n, n))
            out = jax.vmap(inv_single)(flat_y)
            return out.reshape(batch_shape + (n * (n + 1) // 2,))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # Since the transform simply reorders the vector into a matrix, the Jacobian determinant is 1.
        return jnp.zeros(x.shape[:-1])

    @property
    def sign(self):
        # The reordering transformation has a positive derivative everywhere.
        return 1.0

    # Implement tree_flatten and tree_unflatten because base Transform expects them.
    def tree_flatten(self):
        # This transform is stateless.
        return (), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()
