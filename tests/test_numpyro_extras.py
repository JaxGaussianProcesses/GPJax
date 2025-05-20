from jax import (
    grad,
    jit,
)
import jax.numpy as jnp
import numpy as np
import pytest

from gpjax.numpyro_extras import FillTriangularTransform


# Helper function to generate a test input vector for a given matrix size.
def generate_test_vector(n):
    """
    Generate a sequential vector of shape (n(n+1)/2,) with values [1, 2, ..., n(n+1)/2].
    """
    L = n * (n + 1) // 2
    return jnp.arange(1, L + 1, dtype=jnp.float32)


# ----------------- Unit tests using PyTest -----------------


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_forward_inverse(n):
    """
    Test that for a range of input sizes the forward transform correctly fills
    an n x n lower triangular matrix and that the inverse recovers the original vector.
    """
    ft = FillTriangularTransform()
    vec = generate_test_vector(n)
    L = ft(vec)

    # Construct the expected n x n lower triangular matrix
    expected = jnp.zeros((n, n), dtype=vec.dtype)
    row, col = jnp.tril_indices(n)
    expected = expected.at[row, col].set(vec)

    np.testing.assert_allclose(L, expected, rtol=1e-6)

    # Check that the inverse recovers the original vector
    vec_rec = ft.inv(L)
    np.testing.assert_allclose(vec, vec_rec, rtol=1e-6)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_batched_forward_inverse(n):
    """
    Test that the transform correctly handles batched inputs.
    """
    ft = FillTriangularTransform()
    batch_size = 5
    vec = jnp.stack([generate_test_vector(n) for _ in range(batch_size)], axis=0)
    L = ft(vec)  # Expected shape: (batch_size, n, n)
    assert L.shape == (batch_size, n, n)

    vec_rec = ft.inv(L)  # Expected shape: (batch_size, n(n+1)/2)
    assert vec_rec.shape == (batch_size, n * (n + 1) // 2)
    np.testing.assert_allclose(vec, vec_rec, rtol=1e-6)


def test_jit_forward():
    """
    Test that the forward transformation works correctly when compiled with JIT.
    """
    ft = FillTriangularTransform()
    n = 3
    vec = generate_test_vector(n)

    jit_forward = jit(ft)
    L = ft(vec)
    L_jit = jit_forward(vec)
    np.testing.assert_allclose(L, L_jit, rtol=1e-6)


def test_jit_inverse():
    """
    Test that the inverse transformation works correctly when compiled with JIT.
    """
    ft = FillTriangularTransform()
    n = 3
    vec = generate_test_vector(n)
    L_mat = ft(vec)

    # Wrap the inverse call in a lambda to avoid hashing the unhashable _InverseTransform.
    jit_inverse = jit(lambda y: ft.inv(y))
    vec_rec = ft.inv(L_mat)
    vec_rec_jit = jit_inverse(L_mat)
    np.testing.assert_allclose(vec_rec, vec_rec_jit, rtol=1e-6)


def test_grad_forward():
    """
    Test that JAX gradients can be computed for the forward transform.
    We define a simple function that sums the output matrix.
    Since the forward transform is just a reordering, the gradient should be 1
    for every element in the input vector.
    """
    ft = FillTriangularTransform()
    n = 3
    vec = generate_test_vector(n)

    # Define a scalar function f(x) = sum(forward(x))
    f = lambda x: jnp.sum(ft(x))
    grad_f = grad(f)(vec)
    np.testing.assert_allclose(grad_f, jnp.ones_like(vec), rtol=1e-6)


def test_grad_inverse():
    """
    Test that gradients flow through the inverse transformation.
    Define a simple scalar function on the inverse such that g(y) = sum(inv(y)).
    The gradient with respect to y should be one on the lower triangular indices.
    """
    ft = FillTriangularTransform()
    n = 3
    vec = generate_test_vector(n)
    L = ft(vec)

    g = lambda y: jnp.sum(ft.inv(y))
    grad_g = grad(g)(L)

    # Construct the expected gradient matrix: zeros everywhere except ones on the lower triangle.
    grad_expected = jnp.zeros_like(L)
    row, col = jnp.tril_indices(n)
    grad_expected = grad_expected.at[row, col].set(1.0)
    np.testing.assert_allclose(grad_g, grad_expected, rtol=1e-6)


def test_invalid_dimension_error():
    """
    Test that the FillTriangularTransform correctly raises a ValueError when
    the last dimension doesn't equal n(n+1)/2 for some integer n.
    """
    ft = FillTriangularTransform()

    # Create vectors with invalid dimensions that aren't n(n+1)/2 for any integer n
    invalid_dims = [2, 4, 5, 7, 8, 11, 13, 14, 17, 19, 20]

    for dim in invalid_dims:
        vec = jnp.ones(dim)
        with pytest.raises(
            ValueError,
            match="Last dimension must equal n\\(n\\+1\\)/2 for some integer n\\.",
        ):
            ft(vec)

    # Verify that valid dimensions don't raise errors
    valid_dims = [1, 3, 6, 10, 15, 21]  # n(n+1)/2 for n=1,2,3,4,5,6

    for dim in valid_dims:
        vec = jnp.ones(dim)
        try:
            ft(vec)
        except ValueError:
            pytest.fail(
                f"FillTriangularTransform raised ValueError for valid dimension {dim}"
            )


def test_inverse_dimension_error():
    """
    Test that the FillTriangularTransform.inv correctly raises a ValueError when
    the input has less than two dimensions.
    """
    ft = FillTriangularTransform()

    # Create a one-dimensional array
    vec = jnp.ones(3)  # 1D array with 3 elements

    # Try to call inverse on the 1D array, should fail
    with pytest.raises(
        ValueError, match="Input to inverse must be at least two-dimensional."
    ):
        ft.inv(vec)


def test_inverse_non_square_error():
    """
    Test that the FillTriangularTransform.inv correctly raises a ValueError when
    the input matrix is not square.
    """
    ft = FillTriangularTransform()

    # Create non-square matrices of different shapes
    non_square_matrices = [
        jnp.ones((3, 4)),  # 3x4 matrix
        jnp.ones((5, 2)),  # 5x2 matrix
        jnp.ones((1, 3)),  # 1x3 matrix
    ]

    for matrix in non_square_matrices:
        # Extract dimensions
        dim1, dim2 = matrix.shape[-2:]
        # Use a simpler regex pattern that doesn't include parentheses
        error_pattern = "Input matrix must be square; got shape"
        with pytest.raises(ValueError, match=error_pattern):
            ft.inv(matrix)

    # Test with batched non-square matrices
    batched_non_square = jnp.ones((2, 3, 4))  # Batch of 2 matrices of shape 3x4
    with pytest.raises(ValueError, match="Input matrix must be square"):
        ft.inv(batched_non_square)
