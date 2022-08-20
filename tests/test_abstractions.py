from hashlib import new

import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import gpjax as gpx
from gpjax import RBF, Dataset, Gaussian, Prior, initialise, transform
from gpjax.abstractions import InferenceState, fit, fit_batches, get_batch_new_key


@pytest.mark.parametrize("n_iters", [10])
@pytest.mark.parametrize("n", [1, 20])
def test_fit(n_iters, n):
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n, 1)), axis=0)
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    p = Prior(kernel=RBF()) * Gaussian(num_datapoints=n)
    params, trainable_status, constrainer, unconstrainer = initialise(p, key).unpack()
    mll = p.marginal_log_likelihood(D, constrainer, negative=True)
    pre_mll_val = mll(params)
    optimiser = optax.adam(learning_rate=0.1)
    inference_state = fit(mll, params, trainable_status, optimiser, n_iters)
    optimised_params, history = inference_state.params, inference_state.history
    optimised_params = transform(optimised_params, constrainer)
    assert isinstance(inference_state, InferenceState)
    assert isinstance(optimised_params, dict)
    assert mll(optimised_params) < pre_mll_val
    assert isinstance(history, jnp.ndarray)
    assert history.shape[0] == n_iters


def test_stop_grads():
    params = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
    trainables = {"x": True, "y": False}
    loss_fn = lambda params: params["x"] ** 2 + params["y"] ** 2
    optimiser = optax.adam(learning_rate=0.1)
    inference_state = fit(loss_fn, params, trainables, optimiser, n_iters=1)
    learned_params = inference_state.params
    assert isinstance(inference_state, InferenceState)
    assert learned_params["y"] == params["y"]
    assert learned_params["x"] != params["x"]


@pytest.mark.parametrize("n_iters", [5])
@pytest.mark.parametrize("nb", [1, 20, 50])
@pytest.mark.parametrize("ndata", [50])
def test_batch_fitting(n_iters, nb, ndata):
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(ndata, 1)), axis=0)
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    prior = Prior(kernel=RBF())
    likelihood = Gaussian(num_datapoints=ndata)
    p = prior * likelihood
    z = jnp.linspace(-2.0, 2.0, 10).reshape(-1, 1)

    q = gpx.VariationalGaussian(prior=prior, inducing_inputs=z)

    svgp = gpx.StochasticVI(posterior=p, variational_family=q)
    params, trainable_status, constrainer, unconstrainer = initialise(
        svgp, key
    ).unpack()
    params = gpx.transform(params, unconstrainer)
    objective = svgp.elbo(D, constrainer)

    D = Dataset(X=x, y=y)

    optimiser = optax.adam(learning_rate=0.1)
    seed = 42
    print("-" * 80)
    print(params)
    print("-" * 80)
    inference_state = fit_batches(
        objective, params, trainable_status, D, optimiser, seed, nb, n_iters)
    key = jr.PRNGKey(42)
    optimised_params, history = fit_batches(
        objective, params, trainable_status, D, optimiser, key, nb, n_iters)
    optimised_params, history = inference_state.params, inference_state.history
    optimised_params = transform(optimised_params, constrainer)
    assert isinstance(inference_state, InferenceState)
    assert isinstance(optimised_params, dict)
    assert isinstance(history, jnp.ndarray)
    assert history.shape[0] == n_iters


@pytest.mark.parametrize("batch_size", [1, 2, 50])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("ndata", [50])
@pytest.mark.parametrize("key", [jr.PRNGKey(123)])
def test_get_batch_new_key(ndata, ndim, batch_size, key):
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(ndata, ndim)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    B, new_key = get_batch_new_key(D, batch_size, key)

    assert B.n == batch_size
    assert B.X.shape[1:] == x.shape[1:]
    assert B.y.shape[1:] == y.shape[1:]
    assert (new_key != key).all()

    Bnew, _ = get_batch_new_key(D, batch_size, new_key)
    assert Bnew.n == batch_size
    assert Bnew.X.shape[1:] == x.shape[1:]
    assert Bnew.y.shape[1:] == y.shape[1:]
    assert (Bnew.X != B.X).all()
    assert (Bnew.y != B.y).all()
