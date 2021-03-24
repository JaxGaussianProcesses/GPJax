import jax.numpy as jnp
import pytest

from gpjax.types import Dataset, NoneType


def test_nonetype():
    assert isinstance(None, NoneType)


@pytest.mark.parametrize("n", [1, 10, 100])
@pytest.mark.parametrize("outd", [1, 2, 10])
@pytest.mark.parametrize("ind", [1, 2, 10])
def test_dataset(n, outd, ind):
    x = jnp.ones((n, ind))
    y = jnp.ones((n, outd))
    d = Dataset(X=x, y=y)
    assert d.n == n
    assert d.in_dim == ind
    assert d.out_dim == outd


@pytest.mark.parametrize("nx, ny", [(1, 2), (2, 1), (10, 5), (5, 10)])
def test_dataset_assertions(nx, ny):
    x = jnp.ones((nx, 1))
    y = jnp.ones((ny, 1))
    with pytest.raises(AssertionError):
        d = Dataset(X=x, y=y)


def test_y_none():
    x = jnp.ones((10, 1))
    d = Dataset(X=x)
    assert d.y is None
