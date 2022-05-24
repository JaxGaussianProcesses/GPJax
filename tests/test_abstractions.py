import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
import tensorflow as tf

import gpjax as gpx
from gpjax import RBF, Dataset, Gaussian, Prior, initialise, transform
from gpjax.abstractions import fit, fit_batches

tfd = tf.data


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
    optimiser = optax.adam(learning_rate=0.1)
    optimised_params = fit(mll, params, trainable_status, optimiser, n_iters=10)
    optimised_params = transform(optimised_params, constrainer)
    assert isinstance(optimised_params, dict)
    assert mll(optimised_params) < pre_mll_val

def test_stop_grads():
    params = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
    trainables = {"x": True, "y": False}
    loss_fn = lambda params: params["x"] ** 2 + params["y"] ** 2
    optimiser = optax.adam(learning_rate=0.1)
    learned_params = fit(loss_fn, params, trainables, optimiser, n_iters=1)
    assert learned_params["y"] == params["y"]
    assert learned_params["x"] != params["x"]


@pytest.mark.parametrize("nb", [1, 20, 50])
@pytest.mark.parametrize("ndata", [50])
def test_batch_fitting(nb, ndata):
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(ndata, 1)), axis=0)
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    prior = Prior(kernel=RBF())
    likelihood = Gaussian(num_datapoints=ndata)
    p = prior * likelihood
    z = jnp.linspace(-2.0, 2.0, 10).reshape(-1, 1)

    q = gpx.VariationalGaussian(prior=prior, inducing_inputs=z)

    svgp = gpx.SVGP(posterior=p, variational_family=q)
    params, trainable_status, constrainer, unconstrainer = initialise(svgp)
    params = gpx.transform(params, unconstrainer)
    objective = svgp.elbo(D, constrainer)

    D = Dataset(X=x, y=y)
    D = D.cache()
    D = D.repeat()
    D = D.shuffle(D.n)
    D = D.batch(batch_size=nb)
    D = D.prefetch(buffer_size=1)

    optimiser = optax.adam(learning_rate=0.1)
    optimised_params = fit_batches(
        objective, params, trainable_status, D, optimiser, n_iters=5
    )
    optimised_params = transform(optimised_params, constrainer)
    assert isinstance(optimised_params, dict)
