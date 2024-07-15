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

from gpjax.decision_making.test_functions import PoissonTestFunction
from gpjax.typing import KeyArray


@pytest.mark.parametrize("test_function", [PoissonTestFunction()])
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_correct_dtypes(test_function: PoissonTestFunction):
    dataset = test_function.generate_dataset(10, jr.key(42))
    test_x = test_function.generate_test_points(10, jr.key(42))
    assert dataset.X.dtype == jnp.float64
    assert dataset.y.dtype == jnp.integer
    assert test_x.dtype == jnp.float64


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [(PoissonTestFunction(), 1)],
)
@pytest.mark.parametrize("num_samples", [1, 10, 100])
def test_test_points_shape(
    test_function: PoissonTestFunction, dimensionality: int, num_samples: int
):
    test_X = test_function.generate_test_points(num_samples, jr.key(42))
    assert test_X.shape == (num_samples, dimensionality)


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [(PoissonTestFunction(), 1)],
)
@pytest.mark.parametrize("num_samples", [1, 10, 100])
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_dataset_shapes(
    test_function: PoissonTestFunction, dimensionality: int, num_samples: int
):
    dataset = test_function.generate_dataset(num_samples, jr.key(42))
    assert dataset.X.shape == (num_samples, dimensionality)
    assert dataset.y.shape == (num_samples, 1)


@pytest.mark.parametrize("test_function", [PoissonTestFunction()])
@pytest.mark.parametrize("num_samples", [1, 10, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_same_key_same_dataset(
    test_function: PoissonTestFunction, num_samples: int, key: KeyArray
):
    dataset_one = test_function.generate_dataset(num_samples, key)
    dataset_two = test_function.generate_dataset(num_samples, key)
    assert jnp.equal(dataset_one.X, dataset_two.X).all()
    assert jnp.equal(dataset_one.y, dataset_two.y).all()


@pytest.mark.parametrize("test_function", [PoissonTestFunction()])
@pytest.mark.parametrize("num_samples", [10, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_different_key_different_dataset(
    test_function: PoissonTestFunction, num_samples: int, key: KeyArray
):
    dataset_one = test_function.generate_dataset(num_samples, key)
    key, _ = jr.split(key)
    dataset_two = test_function.generate_dataset(num_samples, key)
    assert not jnp.equal(dataset_one.X, dataset_two.X).all()
    assert not jnp.equal(dataset_one.y, dataset_two.y).all()


@pytest.mark.parametrize("test_function", [PoissonTestFunction()])
@pytest.mark.parametrize("num_samples", [1, 10, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_same_key_same_test_points(
    test_function: PoissonTestFunction, num_samples: int, key: KeyArray
):
    test_points_one = test_function.generate_test_points(num_samples, key)
    test_points_two = test_function.generate_test_points(num_samples, key)
    assert jnp.equal(test_points_one, test_points_two).all()


@pytest.mark.parametrize("test_function", [PoissonTestFunction()])
@pytest.mark.parametrize("num_samples", [1, 10, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_different_key_different_test_points(
    test_function: PoissonTestFunction, num_samples: int, key: KeyArray
):
    test_points_one = test_function.generate_test_points(num_samples, key)
    key, _ = jr.split(key)
    test_points_two = test_function.generate_test_points(num_samples, key)
    assert not jnp.equal(test_points_one, test_points_two).all()
