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
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import pytest
import tensorflow_probability.substrates.jax as tfp

from gpjax.decision_making.test_functions import (
    AbstractContinuousTestFunction,
    Forrester,
    LogarithmicGoldsteinPrice,
    Quadratic,
)
from gpjax.typing import KeyArray


def test_abstract_continuous_test_function():
    with pytest.raises(TypeError):
        AbstractContinuousTestFunction()


@pytest.mark.parametrize(
    "test_function", [Quadratic(), Forrester(), LogarithmicGoldsteinPrice()]
)
def test_minimizer_evaluates_to_minimum(test_function: AbstractContinuousTestFunction):
    evaluated_minimizer = test_function.evaluate(test_function.minimizer)
    assert evaluated_minimizer.dtype == jnp.float64
    assert jnp.allclose(
        evaluated_minimizer,
        test_function.minimum,
        atol=1e-5,
    ).all()


@pytest.mark.parametrize(
    "test_function", [Quadratic(), Forrester(), LogarithmicGoldsteinPrice()]
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_correct_minimum(test_function: AbstractContinuousTestFunction, key: KeyArray):
    test_x = test_function.generate_test_points(100000, key)
    function_vals = test_function.evaluate(test_x)
    vals_minus_minimum = function_vals - test_function.minimum
    assert function_vals.dtype == jnp.float64
    assert jnp.all(vals_minus_minimum >= -1e-5)


@pytest.mark.parametrize(
    "test_function", [Quadratic(), Forrester(), LogarithmicGoldsteinPrice()]
)
def test_correct_dtypes(test_function: AbstractContinuousTestFunction):
    minimizer = test_function.minimizer
    minimum = test_function.minimum
    dataset = test_function.generate_dataset(10, jr.key(42))
    test_x = test_function.generate_test_points(10, jr.key(42))
    assert minimizer.dtype == jnp.float64
    assert minimum.dtype == jnp.float64
    assert dataset.X.dtype == jnp.float64
    assert dataset.y.dtype == jnp.float64
    assert test_x.dtype == jnp.float64


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [(Quadratic(), 1), (Forrester(), 1), (LogarithmicGoldsteinPrice(), 2)],
)
@pytest.mark.parametrize("num_samples", [1, 10, 100])
def test_test_points_shape(
    test_function: AbstractContinuousTestFunction, dimensionality: int, num_samples: int
):
    test_X = test_function.generate_test_points(num_samples, jr.key(42))
    assert test_X.shape == (num_samples, dimensionality)


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [(Quadratic(), 1), (Forrester(), 1), (LogarithmicGoldsteinPrice(), 2)],
)
@pytest.mark.parametrize("num_samples", [1, 10, 100])
def test_dataset_shapes(
    test_function: AbstractContinuousTestFunction, dimensionality: int, num_samples: int
):
    dataset = test_function.generate_dataset(num_samples, jr.key(42))
    assert dataset.X.shape == (num_samples, dimensionality)
    assert dataset.y.shape == (num_samples, 1)


@pytest.mark.parametrize(
    "test_function",
    [Quadratic(), Forrester(), LogarithmicGoldsteinPrice()],
)
def test_noiseless_dataset(test_function: AbstractContinuousTestFunction):
    dataset = test_function.generate_dataset(100, jr.key(42))
    expected_y = test_function(dataset.X)
    assert jnp.all(dataset.y == expected_y)


@pytest.mark.parametrize(
    "test_function",
    [Quadratic(), Forrester(), LogarithmicGoldsteinPrice()],
)
@pytest.mark.parametrize("obs_stddev", [1.0, 2.3])
def test_noisy_dataset(
    test_function: AbstractContinuousTestFunction, obs_stddev: float
):
    num_points = 100
    dataset = test_function.generate_dataset(num_points, jr.key(42), obs_stddev)
    noise = tfp.distributions.Normal(
        jnp.zeros(num_points), obs_stddev * jnp.ones(num_points)
    )
    expected_y = test_function(dataset.X) + jnp.transpose(
        noise.sample(sample_shape=[1], seed=jr.key(42))
    )
    assert jnp.all(dataset.y == expected_y)


@pytest.mark.parametrize(
    "test_function", [Quadratic(), Forrester(), LogarithmicGoldsteinPrice()]
)
@pytest.mark.parametrize("num_samples", [1, 10, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_same_key_same_dataset(
    test_function: AbstractContinuousTestFunction, num_samples: int, key: KeyArray
):
    dataset_one = test_function.generate_dataset(num_samples, key)
    dataset_two = test_function.generate_dataset(num_samples, key)
    assert jnp.equal(dataset_one.X, dataset_two.X).all()
    assert jnp.equal(dataset_one.y, dataset_two.y).all()


@pytest.mark.parametrize(
    "test_function", [Quadratic(), Forrester(), LogarithmicGoldsteinPrice()]
)
@pytest.mark.parametrize("num_samples", [1, 10, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_different_key_different_dataset(
    test_function: AbstractContinuousTestFunction, num_samples: int, key: KeyArray
):
    dataset_one = test_function.generate_dataset(num_samples, key)
    key, _ = jr.split(key)
    dataset_two = test_function.generate_dataset(num_samples, key)
    assert not jnp.equal(dataset_one.X, dataset_two.X).all()
    assert not jnp.equal(dataset_one.y, dataset_two.y).all()


@pytest.mark.parametrize(
    "test_function", [Quadratic(), Forrester(), LogarithmicGoldsteinPrice()]
)
@pytest.mark.parametrize("num_samples", [1, 10, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_same_key_same_test_points(
    test_function: AbstractContinuousTestFunction, num_samples: int, key: KeyArray
):
    test_points_one = test_function.generate_test_points(num_samples, key)
    test_points_two = test_function.generate_test_points(num_samples, key)
    assert jnp.equal(test_points_one, test_points_two).all()


@pytest.mark.parametrize(
    "test_function", [Quadratic(), Forrester(), LogarithmicGoldsteinPrice()]
)
@pytest.mark.parametrize("num_samples", [1, 10, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_different_key_different_test_points(
    test_function: AbstractContinuousTestFunction, num_samples: int, key: KeyArray
):
    test_points_one = test_function.generate_test_points(num_samples, key)
    key, _ = jr.split(key)
    test_points_two = test_function.generate_test_points(num_samples, key)
    assert not jnp.equal(test_points_one, test_points_two).all()
