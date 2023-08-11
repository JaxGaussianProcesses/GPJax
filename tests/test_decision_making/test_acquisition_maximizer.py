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

from abc import ABC

from jax.config import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
)
import pytest

from gpjax.decision_making.acquisition_maximizer import (
    AbstractAcquisitionMaximizer,
    ContinuousAcquisitionMaximizer,
    _get_discrete_maximizer,
)
from gpjax.decision_making.search_space import ContinuousSearchSpace
from gpjax.typing import KeyArray

config.update("jax_enable_x64", True)


class TestContinuousAcquisitionFunction(ABC):
    search_space: ContinuousSearchSpace
    maximizer: Float[Array, "1 D"]

    def evaluate(x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        raise NotImplementedError


class NegativeForrester(TestContinuousAcquisitionFunction):
    search_space = ContinuousSearchSpace(
        lower_bounds=jnp.array([0.0], dtype=jnp.float64),
        upper_bounds=jnp.array([1.0], dtype=jnp.float64),
    )
    maximizer = jnp.array([[0.75725]])

    def evaluate(self, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        return -((6 * x - 2) ** 2) * jnp.sin(12 * x - 4)


class NegativeGoldsteinPrice(TestContinuousAcquisitionFunction):
    search_space = ContinuousSearchSpace(
        lower_bounds=jnp.array([-2.0, -2.0], dtype=jnp.float64),
        upper_bounds=jnp.array([2.0, 2.0], dtype=jnp.float64),
    )
    maximizer = jnp.array([[0.0, -1.0]])

    def evaluate(self, x: Float[Array, "N 2"]) -> Float[Array, "N 1"]:
        x1 = x[:, 0]
        x2 = x[:, 1]
        a = 1.0 + (x1 + x2 + 1.0) ** 2 * (
            19.0
            - 14.0 * x1
            + 3.0 * (x1**2)
            - 14.0 * x2
            + 6.0 * x1 * x2
            + 3.0 * (x2**2)
        )
        b = 30.0 + (2.0 * x1 - 3.0 * x2) ** 2 * (
            18.0
            - 32.0 * x1
            + 12.0 * (x1**2)
            + 48.0 * x2
            - 36.0 * x1 * x2
            + 27.0 * (x2**2)
        )
        return -(a * b).reshape(-1, 1)


class Quadratic(TestContinuousAcquisitionFunction):
    search_space = ContinuousSearchSpace(
        lower_bounds=jnp.array([0.0], dtype=jnp.float64),
        upper_bounds=jnp.array([1.0], dtype=jnp.float64),
    )
    maximizer = jnp.array([[0.5]])

    def evaluate(self, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        return -((x - 0.5) ** 2)


def test_abstract_acquisition_maximizer():
    with pytest.raises(TypeError):
        AbstractAcquisitionMaximizer()


@pytest.mark.parametrize(
    "test_acquisition_function, dimensionality",
    [(NegativeForrester(), 1), (NegativeGoldsteinPrice(), 2)],
)
@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10)])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_discrete_maximizer_returns_correct_point(
    test_acquisition_function: TestContinuousAcquisitionFunction,
    dimensionality: int,
    key: KeyArray,
):
    query_points = test_acquisition_function.search_space.sample(1000, key=key)
    acquisition_vals = test_acquisition_function.evaluate(query_points)
    true_max_acquisition_val = jnp.max(acquisition_vals)
    discrete_maximizer = _get_discrete_maximizer(
        query_points, test_acquisition_function.evaluate
    )
    assert discrete_maximizer.shape == (1, dimensionality)
    assert discrete_maximizer.dtype == jnp.float64
    assert (
        test_acquisition_function.evaluate(discrete_maximizer)[0][0]
        == true_max_acquisition_val
    )


@pytest.mark.parametrize("num_initial_samples", [0, -1, -10])
def test_continuous_maximizer_raises_error_with_erroneous_num_initial_samples(
    num_initial_samples: int,
):
    with pytest.raises(ValueError):
        ContinuousAcquisitionMaximizer(num_initial_samples=num_initial_samples)


@pytest.mark.parametrize(
    "test_acquisition_function, dimensionality",
    [(NegativeForrester(), 1), (NegativeGoldsteinPrice(), 2)],
)
@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10)])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_continous_maximizer_returns_same_point_with_same_key(
    test_acquisition_function: TestContinuousAcquisitionFunction,
    dimensionality: int,
    key: KeyArray,
):
    continuous_maximizer_one = ContinuousAcquisitionMaximizer(num_initial_samples=2000)
    continuous_maximizer_two = ContinuousAcquisitionMaximizer(num_initial_samples=2000)
    maximizer_one = continuous_maximizer_one.maximize(
        acquisition_function=test_acquisition_function.evaluate,
        search_space=test_acquisition_function.search_space,
        key=key,
    )
    maximizer_two = continuous_maximizer_two.maximize(
        acquisition_function=test_acquisition_function.evaluate,
        search_space=test_acquisition_function.search_space,
        key=key,
    )
    assert maximizer_one.shape == (1, dimensionality)
    assert maximizer_one.dtype == jnp.float64
    assert maximizer_two.shape == (1, dimensionality)
    assert maximizer_two.dtype == jnp.float64
    assert jnp.equal(maximizer_one, maximizer_two).all()


@pytest.mark.parametrize(
    "test_acquisition_function, dimensionality",
    [
        (NegativeForrester(), 1),
        (NegativeGoldsteinPrice(), 2),
    ],
)
@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10)])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_continuous_maximizer_finds_correct_point(
    test_acquisition_function: TestContinuousAcquisitionFunction,
    dimensionality: int,
    key: KeyArray,
):
    continuous_acquisition_maximizer = ContinuousAcquisitionMaximizer(
        num_initial_samples=1000
    )
    maximizer = continuous_acquisition_maximizer.maximize(
        acquisition_function=test_acquisition_function.evaluate,
        search_space=test_acquisition_function.search_space,
        key=key,
    )
    assert maximizer.shape == (1, dimensionality)
    assert maximizer.dtype == jnp.float64
    assert jnp.allclose(maximizer, test_acquisition_function.maximizer, atol=1e-6).all()


@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10), jr.PRNGKey(1)])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_continuous_maximizer_jaxopt_component(key: KeyArray):
    quadratic_acquisition_function = Quadratic()
    continuous_acquisition_maximizer = ContinuousAcquisitionMaximizer(
        num_initial_samples=1  # Force JaxOpt L-GFBS-B to do the heavy lifting
    )
    maximizer = continuous_acquisition_maximizer.maximize(
        acquisition_function=quadratic_acquisition_function.evaluate,
        search_space=quadratic_acquisition_function.search_space,
        key=key,
    )
    assert maximizer.shape == (1, 1)
    assert maximizer.dtype == jnp.float64
    assert jnp.allclose(
        maximizer, quadratic_acquisition_function.maximizer, atol=1e-6
    ).all()
