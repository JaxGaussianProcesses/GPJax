from hashlib import md5
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from jax.experimental import optimizers
import tensorflow as tf
import typing as tp

from gpjax import RBF, Dataset, Gaussian, Prior, initialise, transform
from gpjax.abstractions import fit, optax_fit, fit_batches, batch_loader

tfd = tf.data

def _dataset_to_tf(dataset, prefetch_buffer=1, batch_size=32):
    X, y, n = dataset.X, dataset.y, dataset.n

    batch_size = min(batch_size, n)

    # Make dataloader, set batch size and prefetch buffer:
    ds = tfd.Dataset.from_tensor_slices((X, y))
    ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(n)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(prefetch_buffer)
    return ds


@pytest.mark.parametrize("n", [20])
def test_fit(n):
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n, 1)), axis=0)
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    p = Prior(kernel=RBF()) * Gaussian(num_datapoints=n)
    params, trainable_status, constrainer, unconstrainer = initialise(p)
    mll = p.marginal_log_likelihood(D, constrainer, negative=True)
    pre_mll_val = mll(params)
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.1)
    optimised_params = fit(
        mll, params, trainable_status, opt_init, opt_update, get_params, n_iters=10
    )
    optimised_params = transform(optimised_params, constrainer)
    assert isinstance(optimised_params, dict)
    assert mll(optimised_params) < pre_mll_val


@pytest.mark.parametrize("n", [20])
def test_optax_fit(n):
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n, 1)), axis=0)
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    p = Prior(kernel=RBF()) * Gaussian(num_datapoints=n)
    params, trainable_status, constrainer, unconstrainer = initialise(p)
    mll = p.marginal_log_likelihood(D, constrainer, negative=True)
    pre_mll_val = mll(params)
    optimiser = optax.adam(learning_rate=0.1)
    optimised_params = optax_fit(mll, params, trainable_status, optimiser, n_iters=10)
    optimised_params = transform(optimised_params, constrainer)
    assert isinstance(optimised_params, dict)
    assert mll(optimised_params) < pre_mll_val


@pytest.mark.parametrize("optim_pkg", ["jax", "optax"])
def test_stop_grads(optim_pkg):
    params = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
    trainables = {"x": True, "y": False}
    loss_fn = lambda params: params["x"] ** 2 + params["y"] ** 2
    if optim_pkg == "jax":
        opt_init, opt_update, get_params = optimizers.adam(step_size=0.1)
        learned_params = fit(
            loss_fn, params, trainables, opt_init, opt_update, get_params, n_iters=1
        )
    elif optim_pkg == "optax":
        optimiser = optax.adam(learning_rate=0.1)
        learned_params = optax_fit(loss_fn, params, trainables, optimiser, n_iters=1)
    assert learned_params["y"] == params["y"]
    assert learned_params["x"] != params["x"]


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n", [50, 100])
def test_batcher(batch_size, n):
    x = jnp.linspace(-3.0, 3.0, num=n).reshape(-1, 1)
    y = jnp.sin(x)
    D = Dataset(X=x, y=y)
    D = _dataset_to_tf(D, batch_size=batch_size)
    batcher = batch_loader(data=D)
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
    D = _dataset_to_tf(D, batch_size=nb)
    batcher = batch_loader(data=D)

    Db = batcher()
    assert Db.X.shape[0] == ndata
    assert isinstance(batcher, tp.Callable)


@pytest.mark.parametrize("nb", [20, 50])
@pytest.mark.parametrize("ndata", [50])
def batch_fitting(nb, ndata):
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(ndata, 1)), axis=0)
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    p = Prior(kernel=RBF()) * Gaussian(num_datapoints=ndata)
    params, trainable_status, constrainer, unconstrainer = initialise(p)
    mll = p.marginal_log_likelihood(D, constrainer, negative=True)
    D = _dataset_to_tf(dataset=D, batch_size=nb)
    batcher = batch_loader(data=D)
    pre_mll_val = mll(params)
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.1)
    optimised_params = fit_batches(
        mll,
        params,
        trainable_status,
        opt_init,
        opt_update,
        get_params,
        get_batch=batcher,
        n_iters=5,
    )
    optimised_params = transform(optimised_params, constrainer)
    assert isinstance(optimised_params, dict)
    assert mll(optimised_params) < pre_mll_val
