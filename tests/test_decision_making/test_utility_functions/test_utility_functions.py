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

from gpjax.decision_making.utility_functions.expected_improvement import (
    ExpectedImprovement,
)

config.update("jax_enable_x64", True)

from beartype.typing import Type
import jax.random as jr
from jaxtyping import TypeCheckError
import pytest

from gpjax.decision_making.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    Forrester,
    LogarithmicGoldsteinPrice,
)
from gpjax.decision_making.utility_functions.base import (
    AbstractSinglePointUtilityFunctionBuilder,
)
from gpjax.decision_making.utility_functions.probability_of_improvement import (
    ProbabilityOfImprovement,
)
from gpjax.decision_making.utility_functions.thompson_sampling import ThompsonSampling
from gpjax.decision_making.utils import OBJECTIVE
from gpjax.typing import KeyArray
from tests.test_decision_making.utils import (
    generate_dummy_conjugate_posterior,
    generate_dummy_non_conjugate_posterior,
)


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
@pytest.mark.parametrize(
    "utility_function_builder, utility_function_kwargs",
    [
        (ExpectedImprovement, {}),
        (ProbabilityOfImprovement, {}),
        (ThompsonSampling, {"num_features": 100}),
    ],
)
def test_utility_function_no_objective_posterior_raises_error(
    utility_function_builder: Type[AbstractSinglePointUtilityFunctionBuilder],
    utility_function_kwargs: dict,
):
    key = jr.key(42)
    forrester = Forrester()
    dataset = forrester.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {"CONSTRAINT": posterior}
    datasets = {OBJECTIVE: dataset}
    with pytest.raises(ValueError):
        utility_function = utility_function_builder(**utility_function_kwargs)
        utility_function.build_utility_function(
            posteriors=posteriors, datasets=datasets, key=key
        )


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
@pytest.mark.parametrize(
    "utility_function_builder, utility_function_kwargs",
    [
        (ExpectedImprovement, {}),
        (ProbabilityOfImprovement, {}),
        (ThompsonSampling, {"num_features": 100}),
    ],
)
def test_utility_function_no_objective_dataset_raises_error(
    utility_function_builder: Type[AbstractSinglePointUtilityFunctionBuilder],
    utility_function_kwargs: dict,
):
    key = jr.key(42)
    forrester = Forrester()
    dataset = forrester.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {"CONSTRAINT": dataset}
    with pytest.raises(ValueError):
        utility_function = utility_function_builder(**utility_function_kwargs)
        utility_function.build_utility_function(
            posteriors=posteriors, datasets=datasets, key=key
        )


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
@pytest.mark.parametrize(
    "utility_function_builder, utility_function_kwargs",
    [
        (ExpectedImprovement, {}),
        (ProbabilityOfImprovement, {}),
        (ThompsonSampling, {"num_features": 100}),
    ],
)
def test_non_conjugate_posterior_raises_error(
    utility_function_builder: Type[AbstractSinglePointUtilityFunctionBuilder],
    utility_function_kwargs: dict,
):
    key = jr.key(42)
    forrester = Forrester()
    dataset = forrester.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_non_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    with pytest.raises(TypeCheckError):
        utility_function = utility_function_builder(**utility_function_kwargs)
        utility_function.build_utility_function(
            posteriors=posteriors, datasets=datasets, key=key
        )


@pytest.mark.parametrize(
    "utility_function_builder, utility_function_kwargs",
    [
        (ExpectedImprovement, {}),
        (ProbabilityOfImprovement, {}),
        (ThompsonSampling, {"num_features": 100}),
    ],
)
@pytest.mark.parametrize(
    "test_target_function",
    [(Forrester()), (LogarithmicGoldsteinPrice())],
)
@pytest.mark.parametrize("num_test_points", [50, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_utility_functions_have_correct_shapes(
    utility_function_builder: Type[AbstractSinglePointUtilityFunctionBuilder],
    utility_function_kwargs: dict,
    test_target_function: AbstractContinuousTestFunction,
    num_test_points: int,
    key: KeyArray,
):
    dataset = test_target_function.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    utility_builder = utility_function_builder(**utility_function_kwargs)
    utility_function = utility_builder.build_utility_function(
        posteriors=posteriors, datasets=datasets, key=key
    )
    test_key, _ = jr.split(key)
    test_X = test_target_function.generate_test_points(num_test_points, test_key)
    utility_function_values = utility_function(test_X)
    assert utility_function_values.shape == (num_test_points, 1)
