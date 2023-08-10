# Copyright 2023 The GPJax Contributors. All Rights Reserved.
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
from beartype.typing import Callable
from jax.config import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
)
import pytest

from gpjax.dataset import Dataset
from gpjax.decision_making.acquisition_functions.thompson_sampling import (
    ThompsonSampling,
)
from gpjax.decision_making.utils import OBJECTIVE
from gpjax.gps import (
    ConjugatePosterior,
    NonConjugatePosterior,
    Prior,
)
from gpjax.kernels import RBF
from gpjax.likelihoods import (
    Gaussian,
    Poisson,
)
from gpjax.mean_functions import Zero

config.update("jax_enable_x64", True)


def generate_dummy_1d_dataset() -> Dataset:
    X = jr.uniform(key=jr.PRNGKey(42), minval=0.0, maxval=10.0, shape=(10, 1))
    y = jnp.sin(X) + X + 1.0 + 10 * jr.normal(jr.PRNGKey(42), shape=X.shape)
    D = Dataset(X, y)
    return D


def generate_dummy_2d_dataset() -> Dataset:
    X = jr.uniform(key=jr.PRNGKey(42), minval=0.0, maxval=10.0, shape=(10, 2))
    X0 = X[:, 0].reshape(-1, 1)
    X1 = X[:, 1].reshape(-1, 1)
    y = (
        jnp.sin(X0)
        + X1
        + X0**2
        + 1.0
        + 10 * jr.normal(jr.PRNGKey(42), shape=X0.shape)
    )
    D = Dataset(X, y)
    return D


def generate_dummy_1d_test_X(num_test_points: int) -> Float[Array, "N 1"]:
    test_X = jnp.linspace(0.0, 10.0, num_test_points).reshape(-1, 1)
    return test_X


def generate_dummy_2d_test_X(sqrt_num_test_points: int) -> Float[Array, "N 2"]:
    test_X0 = jnp.linspace(0.0, 10.0, sqrt_num_test_points)
    test_X1 = jnp.linspace(0.0, 10.0, sqrt_num_test_points)
    test_X0, test_X1 = jnp.meshgrid(test_X0, test_X1)
    test_X = jnp.column_stack((test_X0.ravel(), test_X1.ravel()))
    return test_X


def generate_dummy_conjugate_posterior(dataset: Dataset) -> ConjugatePosterior:
    kernel = RBF(lengthscale=jnp.ones(dataset.X.shape[1]))
    mean_function = Zero()
    prior = Prior(kernel=kernel, mean_function=mean_function)
    likelihood = Gaussian(num_datapoints=dataset.n)
    posterior = prior * likelihood
    return posterior


def generate_dummy_non_conjugate_posterior(dataset: Dataset) -> NonConjugatePosterior:
    kernel = RBF(lengthscale=jnp.ones(dataset.X.shape[1]))
    mean_function = Zero()
    prior = Prior(kernel=kernel, mean_function=mean_function)
    likelihood = Poisson(num_datapoints=dataset.n)
    posterior = prior * likelihood
    return posterior


def test_thompson_sampling_no_objective_posterior_raises_error():
    dataset = generate_dummy_1d_dataset()
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {"CONSTRAINT": posterior}
    datasets = {OBJECTIVE: dataset}
    key = jr.PRNGKey(42)
    with pytest.raises(ValueError):
        ts_acquisition_builder = ThompsonSampling(num_rff_features=500)
        ts_acquisition_builder.build_acquisition_function(
            posteriors=posteriors, datasets=datasets, key=key
        )


def test_thompson_sampling_no_objective_dataset_raises_error():
    dataset = generate_dummy_1d_dataset()
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {"CONSTRAINT": dataset}
    key = jr.PRNGKey(42)
    with pytest.raises(ValueError):
        ts_acquisition_builder = ThompsonSampling(num_rff_features=500)
        ts_acquisition_builder.build_acquisition_function(
            posteriors=posteriors, datasets=datasets, key=key
        )


def test_thompson_sampling_none_key_raises_error():
    dataset = generate_dummy_1d_dataset()
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    with pytest.raises(ValueError):
        ts_acquisition_builder = ThompsonSampling(num_rff_features=500)
        ts_acquisition_builder.build_acquisition_function(
            posteriors=posteriors, datasets=datasets, key=None
        )


def test_thompson_sampling_non_conjugate_posterior_raises_error():
    dataset = generate_dummy_1d_dataset()
    posterior = generate_dummy_non_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    key = jr.PRNGKey(42)
    with pytest.raises(ValueError):
        ts_acquisition_builder = ThompsonSampling(num_rff_features=500)
        ts_acquisition_builder.build_acquisition_function(
            posteriors=posteriors, datasets=datasets, key=key
        )


@pytest.mark.parametrize(
    "dataset, dimensionality",
    [(generate_dummy_1d_dataset(), 1), (generate_dummy_2d_dataset(), 2)],
)
@pytest.mark.parametrize("num_test_points", [49, 100])
def test_thompson_sampling_acquisition_function_correct_shapes(
    dataset: Dataset, dimensionality: int, num_test_points: int
):
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    key = jr.PRNGKey(42)
    ts_acquisition_builder = ThompsonSampling(num_rff_features=500)
    ts_acquisition_function = ts_acquisition_builder.build_acquisition_function(
        posteriors=posteriors, datasets=datasets, key=key
    )
    if dimensionality == 1:
        test_X = generate_dummy_1d_test_X(num_test_points)
    elif dimensionality == 2:
        test_X = generate_dummy_2d_test_X(int(jnp.sqrt(num_test_points)))
    ts_acquisition_function_values = ts_acquisition_function(test_X)
    assert ts_acquisition_function_values.shape == (num_test_points, 1)


def test_thompson_sampling_acquisition_function_same_key_same_function():
    dataset = generate_dummy_1d_dataset()
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    key = jr.PRNGKey(42)
    ts_acquisition_builder_one = ThompsonSampling(num_rff_features=500)
    ts_acquisition_builder_two = ThompsonSampling(num_rff_features=500)
    ts_acquisition_function_one = ts_acquisition_builder_one.build_acquisition_function(
        posteriors=posteriors, datasets=datasets, key=key
    )
    ts_acquisition_function_two = ts_acquisition_builder_two.build_acquisition_function(
        posteriors=posteriors, datasets=datasets, key=key
    )
    test_X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    ts_acquisition_function_one_values = ts_acquisition_function_one(test_X)
    ts_acquisition_function_two_values = ts_acquisition_function_two(test_X)
    assert isinstance(ts_acquisition_function_one, Callable)
    assert isinstance(ts_acquisition_function_two, Callable)
    assert (
        ts_acquisition_function_one_values == ts_acquisition_function_two_values
    ).all()


def test_thompson_sampling_acquisition_function_different_key_different_function():
    dataset = generate_dummy_1d_dataset()
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    key_one = jr.PRNGKey(42)
    key_two = jr.PRNGKey(43)
    ts_acquisition_builder = ThompsonSampling(num_rff_features=500)
    ts_acquisition_function_one = ts_acquisition_builder.build_acquisition_function(
        posteriors=posteriors, datasets=datasets, key=key_one
    )
    ts_acquisition_function_two = ts_acquisition_builder.build_acquisition_function(
        posteriors=posteriors, datasets=datasets, key=key_two
    )
    test_X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    ts_acquisition_function_one_values = ts_acquisition_function_one(test_X)
    ts_acquisition_function_two_values = ts_acquisition_function_two(test_X)
    assert isinstance(ts_acquisition_function_one, Callable)
    assert isinstance(ts_acquisition_function_two, Callable)
    assert not (
        ts_acquisition_function_one_values == ts_acquisition_function_two_values
    ).all()
