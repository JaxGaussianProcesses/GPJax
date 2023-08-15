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
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax.decision_making.acquisition_maximizer import (
    AbstractAcquisitionMaximizer,
    ContinuousAcquisitionMaximizer,
    _get_discrete_maximizer,
)
from gpjax.decision_making.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    Forrester,
    LogarithmicGoldsteinPrice,
    Quadratic,
)
from gpjax.typing import KeyArray


def test_abstract_acquisition_maximizer():
    with pytest.raises(TypeError):
        AbstractAcquisitionMaximizer()


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [(Forrester(), 1), (LogarithmicGoldsteinPrice(), 2)],
)
@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10)])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_discrete_maximizer_returns_correct_point(
    test_function: AbstractContinuousTestFunction,
    dimensionality: int,
    key: KeyArray,
):
    query_points = test_function.search_space.sample(1000, key=key)
    acquisition_function = lambda x: -1.0 * test_function.evaluate(x)
    acquisition_vals = acquisition_function(query_points)
    true_max_acquisition_val = jnp.max(acquisition_vals)
    discrete_maximizer = _get_discrete_maximizer(query_points, acquisition_function)
    assert discrete_maximizer.shape == (1, dimensionality)
    assert discrete_maximizer.dtype == jnp.float64
    assert acquisition_function(discrete_maximizer)[0][0] == true_max_acquisition_val


@pytest.mark.parametrize("num_initial_samples", [0, -1, -10])
def test_continuous_maximizer_raises_error_with_erroneous_num_initial_samples(
    num_initial_samples: int,
):
    with pytest.raises(ValueError):
        ContinuousAcquisitionMaximizer(
            num_initial_samples=num_initial_samples, num_restarts=1
        )


@pytest.mark.parametrize("num_restarts", [0, -1, -10])
def test_continuous_maximizer_raises_error_with_erroneous_num_restarts(
    num_restarts: int,
):
    with pytest.raises(ValueError):
        ContinuousAcquisitionMaximizer(num_initial_samples=1, num_restarts=num_restarts)


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [(Forrester(), 1), (LogarithmicGoldsteinPrice(), 2)],
)
@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10)])
@pytest.mark.parametrize("num_restarts", [1, 3])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_continous_maximizer_returns_same_point_with_same_key(
    test_function: AbstractContinuousTestFunction,
    dimensionality: int,
    key: KeyArray,
    num_restarts: int,
):
    continuous_maximizer_one = ContinuousAcquisitionMaximizer(
        num_initial_samples=1000, num_restarts=num_restarts
    )
    continuous_maximizer_two = ContinuousAcquisitionMaximizer(
        num_initial_samples=1000, num_restarts=num_restarts
    )
    acquisition_function = lambda x: -1.0 * test_function.evaluate(x)
    maximizer_one = continuous_maximizer_one.maximize(
        acquisition_function=acquisition_function,
        search_space=test_function.search_space,
        key=key,
    )
    maximizer_two = continuous_maximizer_two.maximize(
        acquisition_function=acquisition_function,
        search_space=test_function.search_space,
        key=key,
    )
    assert maximizer_one.shape == (1, dimensionality)
    assert maximizer_one.dtype == jnp.float64
    assert maximizer_two.shape == (1, dimensionality)
    assert maximizer_two.dtype == jnp.float64
    assert jnp.equal(maximizer_one, maximizer_two).all()


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [
        (Forrester(), 1),
        (LogarithmicGoldsteinPrice(), 2),
    ],
)
@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10)])
@pytest.mark.parametrize("num_restarts", [1, 3])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_continuous_maximizer_finds_correct_point(
    test_function: AbstractContinuousTestFunction,
    dimensionality: int,
    key: KeyArray,
    num_restarts: int,
):
    continuous_acquisition_maximizer = ContinuousAcquisitionMaximizer(
        num_initial_samples=1000, num_restarts=num_restarts
    )
    acquisition_function = lambda x: -1.0 * test_function.evaluate(x)
    true_acquisition_maximizer = test_function.minimizer
    maximizer = continuous_acquisition_maximizer.maximize(
        acquisition_function=acquisition_function,
        search_space=test_function.search_space,
        key=key,
    )
    assert maximizer.shape == (1, dimensionality)
    assert maximizer.dtype == jnp.float64
    assert jnp.allclose(maximizer, true_acquisition_maximizer, atol=1e-6).all()


@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10), jr.PRNGKey(1)])
@pytest.mark.parametrize("num_restarts", [1, 3])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_continuous_maximizer_jaxopt_component(key: KeyArray, num_restarts: int):
    quadratic = Quadratic()
    continuous_acquisition_maximizer = ContinuousAcquisitionMaximizer(
        num_initial_samples=1,  # Force JaxOpt L-GFBS-B to do the heavy lifting
        num_restarts=num_restarts,
    )
    acquisition_function = lambda x: -1.0 * quadratic.evaluate(x)
    true_acquisition_maximizer = quadratic.minimizer
    maximizer = continuous_acquisition_maximizer.maximize(
        acquisition_function=acquisition_function,
        search_space=quadratic.search_space,
        key=key,
    )
    assert maximizer.shape == (1, 1)
    assert maximizer.dtype == jnp.float64
    assert jnp.allclose(maximizer, true_acquisition_maximizer, atol=1e-6).all()
