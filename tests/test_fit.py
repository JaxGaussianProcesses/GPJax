# Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
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


from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
import optax as ox
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax.config import config

from gpjax.base import Module, param_field
from gpjax.dataset import Dataset
from gpjax.fit import fit, get_batch
from gpjax.gps import ConjugatePosterior, Prior
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant
from gpjax.objectives import ELBO, AbstractObjective, ConjugateMLL
from gpjax.variational_families import VariationalGaussian

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_simple_linear_model() -> None:
    # Create dataset:
    X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    y = 2.0 * X + 1.0 + 10 * jr.normal(jr.PRNGKey(0), X.shape).reshape(-1, 1)
    D = Dataset(X, y)

    # Define linear model:
    @dataclass
    class LinearModel(Module):
        weight: float = param_field(bijector=tfb.Identity())
        bias: float = param_field(bijector=tfb.Identity(), trainable=False)

        def __call__(self, x):
            return self.weight * x + self.bias

    model = LinearModel(weight=1.0, bias=1.0)

    # Define loss function:
    @dataclass
    class MeanSqaureError(AbstractObjective):
        def step(self, model: LinearModel, train_data: Dataset) -> float:
            return jnp.mean((train_data.y - model(train_data.X)) ** 2)

    loss = MeanSqaureError()

    # Train!
    trained_model, hist = fit(
        model=model, objective=loss, train_data=D, optim=ox.sgd(0.001), num_iters=100
    )

    # Ensure we return a history of the correct length
    assert len(hist) == 100

    # Ensure we return a model of the same class
    assert isinstance(trained_model, LinearModel)

    # Test reduction in loss:
    assert loss(trained_model, D) < loss(model, D)

    # Test stop_gradient on bias:
    assert trained_model.bias == 1.0


@pytest.mark.parametrize("num_iters", [1, 5])
@pytest.mark.parametrize("n_data", [1, 20])
@pytest.mark.parametrize("verbose", [True, False])
def test_gaussian_process_regression(num_iters, n_data: int, verbose: bool) -> None:
    # Create dataset:
    key = jr.PRNGKey(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, 1)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define GP model:
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = Gaussian(num_datapoints=n_data)
    posterior = prior * likelihood

    # Define loss function:
    mll = ConjugateMLL(negative=True)

    # Train!
    trained_model, history = fit(
        model=posterior,
        objective=mll,
        train_data=D,
        optim=ox.adam(0.1),
        num_iters=num_iters,
        verbose=verbose,
    )

    # Ensure the trained model is a Gaussian process posterior
    assert isinstance(trained_model, ConjugatePosterior)

    # Ensure we return a history of the correct length
    assert len(history) == num_iters

    # Ensure we reduce the loss
    assert mll(trained_model, D) < mll(posterior, D)


@pytest.mark.parametrize("num_iters", [1, 5])
@pytest.mark.parametrize("batch_size", [1, 20, 50])
@pytest.mark.parametrize("n_data", [50])
@pytest.mark.parametrize("verbose", [True, False])
def test_batch_fitting(
    num_iters: int, batch_size: int, n_data: int, verbose: bool
) -> None:
    # Create dataset:
    key = jr.PRNGKey(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, 1)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define GP model:
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = Gaussian(num_datapoints=n_data)
    posterior = prior * likelihood

    # Define variational family:
    z = jnp.linspace(-2.0, 2.0, 10).reshape(-1, 1)
    q = VariationalGaussian(posterior=posterior, inducing_inputs=z)

    # Define loss function:
    elbo = ELBO(negative=True)

    # Train!
    trained_model, history = fit(
        model=q,
        objective=elbo,
        train_data=D,
        optim=ox.adam(0.1),
        num_iters=num_iters,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Ensure the trained model is a Gaussian process posterior
    assert isinstance(trained_model, VariationalGaussian)

    # Ensure we return a history of the correct length
    assert len(history) == num_iters

    # Ensure we reduce the loss
    assert elbo(trained_model, D) < elbo(q, D)


@pytest.mark.parametrize("n_data", [50])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 2, 50])
def test_get_batch(n_data: int, n_dim: int, batch_size: int):
    key = jr.PRNGKey(123)

    # Create dataset:
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, n_dim)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Sample out a batch:
    B = get_batch(D, batch_size, key)

    # Check batch is correct size and shape dimensions:
    assert B.n == batch_size
    assert B.X.shape[1:] == x.shape[1:]
    assert B.y.shape[1:] == y.shape[1:]

    # Ensure no caching of batches:
    key, subkey = jr.split(key)
    New = get_batch(D, batch_size, subkey)
    assert New.n == batch_size
    assert New.X.shape[1:] == x.shape[1:]
    assert New.y.shape[1:] == y.shape[1:]
    assert (New.X != B.X).all()
    assert (New.y != B.y).all()
