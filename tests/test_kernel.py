import numpy as np
from gpblocks.types import SquaredExponential
from gpblocks.kernel import compute_gram
import pytest


def _is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def _looped_gram_sqexp(x, y, lengthscale, variance):
    K = np.empty((x.shape[0], y.shape[0]))
    K[:] = np.nan
    for idx, a in enumerate(x):
        for jdx, b in enumerate(y):
            tau = np.square(np.abs(a-b))
            K[idx, jdx] = variance**2 * np.exp(-tau/(2*lengthscale**2))
    return K


@pytest.mark.parametrize("lengthscale, variance", [[0.5, 1.1]])
def test_gram(lengthscale, variance):
    x = np.linspace(-1, 1, num=100).reshape(-1, 1)
    kernel = SquaredExponential(lengthscale=[lengthscale], variance=[variance])
    gram_matrix = compute_gram(kernel, x, x)
    exact_gram = _looped_gram_sqexp(x, x, lengthscale, variance)
    np.testing.assert_allclose(exact_gram, gram_matrix)


@pytest.mark.parametrize("jitter", [[1e-6]])
def test_positive_definite(jitter):
    x = np.linspace(0, 0.1, 300).reshape(-1, 1)  # Points close enough to make Cholesky unstable
    kernel = SquaredExponential(lengthscale=[0.1], variance=[0.5])
    # gram = compute_gram(kernel, x, x)  # This is definitely not PSD
    stable_gram = compute_gram(kernel, x, x, jitter)  # To pass the test, this should be PSD
    assert _is_pos_def(stable_gram)
