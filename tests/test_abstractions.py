import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from jax.experimental import optimizers

from gpjax import RBF, Dataset, Gaussian, Prior, initialise, transform
from gpjax.abstractions import fit, optax_fit


@pytest.mark.parametrize("n", [20])
def test_fit(n):
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n, 1)), axis=0)
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    p = Prior(kernel=RBF()) * Gaussian(num_datapoints=n)
    params, constrainer, unconstrainer = initialise(p)
    mll = p.marginal_log_likelihood(D, constrainer, negative=True)
    pre_mll_val = mll(params)
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.1)
    optimised_params = fit(mll, params, opt_init, opt_update, get_params, n_iters=10)
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
    params, constrainer, unconstrainer = initialise(p)
    mll = p.marginal_log_likelihood(D, constrainer, negative=True)
    pre_mll_val = mll(params)
    optimiser = optax.adam(learning_rate=0.1)
    optimised_params = optax_fit(mll, params, optimiser, n_iters=10)
    optimised_params = transform(optimised_params, constrainer)
    assert isinstance(optimised_params, dict)
    assert mll(optimised_params) < pre_mll_val
