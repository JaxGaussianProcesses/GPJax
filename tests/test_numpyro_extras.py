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
