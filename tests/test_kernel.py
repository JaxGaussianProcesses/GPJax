import jax.numpy as jnp
from gpblocks.kernel import gram, rbf, jitter_matrix
import pytest


@pytest.mark.parametrize("lengthscale, variance", [[0.5, 1.1]])
def test_gram(lengthscale, variance):
    """
    Test if the Gram matrix produced by the RBF kernel function is symmetric positive-definite.
    :param lengthscale:
    :param variance:
    :return:
    """
    x = jnp.linspace(-1., 1., num=100)
    kernel = rbf(lengthscale, variance)
    jitter = jitter_matrix(x.shape[0], 1e-6)
    K = gram(kernel, x) + jitter
    print(jnp.linalg.eigvals(K))
    assert _is_pos_def(K), "Gram matrix is not positive-definite"
    assert _is_symmetric(K), "Gram matrix is not symmetric"


def _is_pos_def(x):
    return jnp.all(jnp.linalg.eigvals(x) > 0)


def _is_symmetric(A, rtol=1e-05, atol=1e-08):
    return jnp.allclose(A, A.T, rtol=rtol, atol=atol)


#
# def _looped_gram_sqexp(x, y, lengthscale, variance):
#     K = np.empty((x.shape[0], y.shape[0]))
#     K[:] = np.nan
#     for idx, a in enumerate(x):
#         for jdx, b in enumerate(y):
#             tau = np.square(np.abs(a-b))
#             K[idx, jdx] = variance**2 * np.exp(-tau/(2*lengthscale**2))
#     return K
#
#
# @pytest.mark.parametrize("lengthscale, variance", [[0.5, 1.1]])
# def test_gram(lengthscale, variance):
#     x = np.linspace(-1, 1, num=100).reshape(-1, 1)
#     kernel = SquaredExponential(lengthscale=[lengthscale], variance=[variance])
#     gram_matrix = compute_gram(kernel, x, x)
#     exact_gram = _looped_gram_sqexp(x, x, lengthscale, variance)
#     np.testing.assert_allclose(exact_gram, gram_matrix)
#
#
# @pytest.mark.parametrize("jitter", [[1e-6]])
# def test_positive_definite(jitter):
#     x = np.linspace(0, 0.1, 300).reshape(-1, 1)  # Points close enough to make Cholesky unstable
#     kernel = SquaredExponential(lengthscale=[0.1], variance=[0.5])
#     # gram = compute_gram(kernel, x, x)  # This is definitely not PSD
#     stable_gram = compute_gram(kernel, x, x, jitter)  # To pass the test, this should be PSD
#     assert _is_pos_def(stable_gram)
