import jax.numpy as jnp
import pytest
import typing as tp

from gpjax.types import Dataset, NoneType, verify_dataset


def test_nonetype():
    assert isinstance(None, NoneType)


@pytest.mark.parametrize("n", [1, 10, 100])
@pytest.mark.parametrize("outd", [1, 2, 10])
@pytest.mark.parametrize("ind", [1, 2, 10])
def test_dataset(n, outd, ind):
    x = jnp.ones((n, ind))
    y = jnp.ones((n, outd))
    d = Dataset(X=x, y=y)
    verify_dataset(d)
    assert d.n == n
    assert d.in_dim == ind
    assert d.out_dim == outd


@pytest.mark.parametrize("nx, ny", [(1, 2), (2, 1), (10, 5), (5, 10)])
def test_dataset_assertions(nx, ny):
    x = jnp.ones((nx, 1))
    y = jnp.ones((ny, 1))
    with pytest.raises(AssertionError):
        ds = Dataset(X=x, y=y)
        verify_dataset(ds)


def test_y_none():
    x = jnp.ones((10, 1))
    d = Dataset(X=x)
    verify_dataset(d)
    assert d.y is None


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n", [50, 100])
def test_batcher(batch_size, n):
    x = jnp.linspace(-3.0, 3.0, num=n).reshape(-1, 1)
    y = jnp.sin(x)
    D = Dataset(X=x, y=y)
    D = D.cache()
    D = D.repeat()
    D = D.shuffle(D.n)
    D = D.batch(batch_size)
    D = D.prefetch(buffer_size=1)
    batcher = D.get_batcher()
    Db = batcher()
    assert Db.X.shape[0] == batch_size
    assert Db.y.shape[0] == batch_size
    assert Db.n == batch_size
    assert isinstance(Db, Dataset)

    Db2 = batcher()
    assert any(Db2.X != Db.X)
    assert any(Db2.y != Db.y)
    assert Db2.n == batch_size
    assert isinstance(Db2, Dataset)


@pytest.mark.parametrize("nb", [20, 50])
@pytest.mark.parametrize("ndata", [10])
def test_min_batch(nb, ndata):
    x = jnp.linspace(-3.0, 3.0, num=ndata).reshape(-1, 1)
    y = jnp.sin(x)
    D = Dataset(X=x, y=y)
    D = D.batch(batch_size=nb)
    batcher = D.get_batcher()

    Db = batcher()
    assert Db.X.shape[0] == ndata
    assert isinstance(batcher, tp.Callable)
