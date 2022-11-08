# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from jax.config import config

import gpjax as gpx
from gpjax import RBF, Dataset, Gaussian, Prior, initialise
from gpjax.abstractions import InferenceState, fit, fit_batches, fit_natgrads, get_batch
from gpjax.parameters import ParameterState, build_bijectors

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


@pytest.mark.parametrize("n_iters", [1, 5])
@pytest.mark.parametrize("n", [1, 20])
@pytest.mark.parametrize("verbose", [True, False])
def test_fit(n_iters, n, verbose):
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n, 1)), axis=0)
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    p = Prior(kernel=RBF()) * Gaussian(num_datapoints=n)
    parameter_state = initialise(p, key)
    mll = p.marginal_log_likelihood(D, negative=True)
    pre_mll_val = mll(parameter_state.params)
    optimiser = optax.adam(learning_rate=0.1)
    inference_state = fit(mll, parameter_state, optimiser, n_iters, verbose=verbose)
    optimised_params, history = inference_state.unpack()
    assert isinstance(inference_state, InferenceState)
    assert isinstance(optimised_params, dict)
    assert mll(optimised_params) < pre_mll_val
    assert isinstance(history, jnp.ndarray)
    assert history.shape[0] == n_iters


def test_stop_grads():
    params = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
    trainables = {"x": True, "y": False}
    bijectors = build_bijectors(params)

    def loss_fn(params):
        return params["x"] ** 2 + params["y"] ** 2

    optimiser = optax.adam(learning_rate=0.1)
    parameter_state = ParameterState(
        params=params, trainables=trainables, bijectors=bijectors
    )
    inference_state = fit(loss_fn, parameter_state, optimiser, n_iters=1)
    learned_params = inference_state.params
    assert isinstance(inference_state, InferenceState)
    assert learned_params["y"] == params["y"]
    assert learned_params["x"] != params["x"]


@pytest.mark.parametrize("n_iters", [1, 5])
@pytest.mark.parametrize("nb", [1, 20, 50])
@pytest.mark.parametrize("ndata", [50])
@pytest.mark.parametrize("verbose", [True, False])
def test_batch_fitting(n_iters, nb, ndata, verbose):
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
    parameter_state = initialise(svgp, key)
    objective = svgp.elbo(D, negative=True)

    pre_mll_val = objective(parameter_state.params, D)

    D = Dataset(X=x, y=y)

    optimiser = optax.adam(learning_rate=0.1)
    key = jr.PRNGKey(42)
    inference_state = fit_batches(
        objective, parameter_state, D, optimiser, key, nb, n_iters, verbose=verbose
    )
    optimised_params, history = inference_state.unpack()
    assert isinstance(inference_state, InferenceState)
    assert isinstance(optimised_params, dict)
    assert objective(optimised_params, D) < pre_mll_val
    assert isinstance(history, jnp.ndarray)
    assert history.shape[0] == n_iters


@pytest.mark.parametrize("n_iters", [1, 5])
@pytest.mark.parametrize("nb", [1, 20, 50])
@pytest.mark.parametrize("ndata", [50])
@pytest.mark.parametrize("verbose", [True, False])
def test_natural_gradients(ndata, nb, n_iters, verbose):
    key = jr.PRNGKey(123)
    x = jnp.sort(jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(ndata, 1)), axis=0)
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)
    prior = Prior(kernel=RBF())
    likelihood = Gaussian(num_datapoints=ndata)
    p = prior * likelihood
    z = jnp.linspace(-2.0, 2.0, 10).reshape(-1, 1)

    q = gpx.NaturalVariationalGaussian(prior=prior, inducing_inputs=z)

    svgp = gpx.StochasticVI(posterior=p, variational_family=q)
    training_state = initialise(svgp, key)

    D = Dataset(X=x, y=y)

    hyper_optimiser = optax.adam(learning_rate=0.1)
    moment_optimiser = optax.sgd(learning_rate=1.0)

    objective = svgp.elbo(D, negative=True)
    parameter_state = initialise(svgp, key)
    pre_mll_val = objective(parameter_state.params, D)

    key = jr.PRNGKey(42)
    inference_state = fit_natgrads(
        svgp,
        training_state,
        D,
        moment_optimiser,
        hyper_optimiser,
        key,
        nb,
        n_iters,
        verbose=verbose,
    )
    optimised_params, history = inference_state.unpack()
    assert isinstance(inference_state, InferenceState)
    assert isinstance(optimised_params, dict)
    assert objective(optimised_params, D) < pre_mll_val
    assert isinstance(history, jnp.ndarray)
    assert history.shape[0] == n_iters


@pytest.mark.parametrize("batch_size", [1, 2, 50])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("ndata", [50])
@pytest.mark.parametrize("key", [jr.PRNGKey(123)])
def test_get_batch(ndata, ndim, batch_size, key):
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(ndata, ndim)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    B = get_batch(D, batch_size, key)

    assert B.n == batch_size
    assert B.X.shape[1:] == x.shape[1:]
    assert B.y.shape[1:] == y.shape[1:]

    # test no caching of batches:
    key, subkey = jr.split(key)
    Bnew = get_batch(D, batch_size, subkey)
    assert Bnew.n == batch_size
    assert Bnew.X.shape[1:] == x.shape[1:]
    assert Bnew.y.shape[1:] == y.shape[1:]
    assert (Bnew.X != B.X).all()
    assert (Bnew.y != B.y).all()
