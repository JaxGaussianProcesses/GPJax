import jax.numpy as jnp
import pytest

from gpjax.mean_functions import Zero, initialise


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_shape(dim):
    x = jnp.linspace(-1.0, 1.0, num=10).reshape(-1, 1)
    if dim > 1:
        x = jnp.hstack([x] * dim)
    meanf = Zero()
    mu = meanf(x)
    assert mu.shape[0] == x.shape[0]
    assert mu.shape[1] == 1


def test_initialisers():
    params = initialise(Zero())
    assert not params
