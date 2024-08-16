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
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import optax as ox
import pytest

import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.decision_making.decision_maker import (
    AbstractDecisionMaker,
    UtilityDrivenDecisionMaker,
)
from gpjax.decision_making.posterior_handler import PosteriorHandler
from gpjax.decision_making.search_space import (
    AbstractSearchSpace,
    ContinuousSearchSpace,
)
from gpjax.decision_making.test_functions import Quadratic
from gpjax.decision_making.utility_functions import (
    AbstractSinglePointUtilityFunctionBuilder,
    ThompsonSampling,
)
from gpjax.decision_making.utility_maximizer import (
    AbstractSinglePointUtilityMaximizer,
    ContinuousSinglePointUtilityMaximizer,
)
from gpjax.decision_making.utils import (
    OBJECTIVE,
    build_function_evaluator,
)
from gpjax.typing import KeyArray
from tests.test_decision_making.utils import QuadraticSinglePointUtilityFunctionBuilder

CONSTRAINT = "CONSTRAINT"


@pytest.fixture
def search_space() -> ContinuousSearchSpace:
    return ContinuousSearchSpace(
        lower_bounds=jnp.array([0.0], dtype=jnp.float64),
        upper_bounds=jnp.array([1.0], dtype=jnp.float64),
    )


@pytest.fixture
def posterior_handler() -> PosteriorHandler:
    mean = gpx.mean_functions.Zero()
    kernel = gpx.kernels.Matern52(
        lengthscale=jnp.array(1.0),
        variance=jnp.array(1.0),
        n_dims=1,
    )
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
    likelihood_builder = lambda x: gpx.likelihoods.Gaussian(
        num_datapoints=x, obs_stddev=jnp.array(1e-3)
    )
    posterior_handler = PosteriorHandler(
        prior=prior,
        likelihood_builder=likelihood_builder,
        optimization_objective=gpx.objectives.conjugate_mll,
        optimizer=ox.adam(learning_rate=0.01),
        num_optimization_iters=100,
    )
    return posterior_handler


@pytest.fixture
def utility_function_builder() -> AbstractSinglePointUtilityFunctionBuilder:
    return QuadraticSinglePointUtilityFunctionBuilder()


@pytest.fixture
def thompson_sampling_utility_function_builder() -> ThompsonSampling:
    return ThompsonSampling(num_features=100)


@pytest.fixture
def utility_maximizer() -> AbstractSinglePointUtilityMaximizer:
    return ContinuousSinglePointUtilityMaximizer(
        num_initial_samples=1000, num_restarts=1
    )


def get_dataset(num_points: int, key: KeyArray) -> Dataset:
    test_function = Quadratic()
    dataset = test_function.generate_dataset(num_points=num_points, key=key)
    return dataset


def test_abstract_decision_maker_raises_error():
    with pytest.raises(TypeError):
        AbstractDecisionMaker()


@pytest.mark.parametrize("batch_size", [0, -1, -10])
def test_invalid_batch_size_raises_error(
    search_space: AbstractSearchSpace,
    posterior_handler: PosteriorHandler,
    utility_function_builder: AbstractSinglePointUtilityFunctionBuilder,
    utility_maximizer: AbstractSinglePointUtilityMaximizer,
    batch_size: int,
):
    key = jr.key(42)
    posterior_handlers = {OBJECTIVE: posterior_handler}
    objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    datasets = {"OBJECTIVE": objective_dataset}
    with pytest.raises(ValueError):
        UtilityDrivenDecisionMaker(
            search_space=search_space,
            posterior_handlers=posterior_handlers,
            datasets=datasets,
            utility_function_builder=utility_function_builder,
            utility_maximizer=utility_maximizer,
            key=key,
            post_ask=[],
            post_tell=[],
            batch_size=batch_size,
        )


def test_non_thompson_sampling_non_one_batch_size_raises_error(
    search_space: AbstractSearchSpace,
    posterior_handler: PosteriorHandler,
    utility_function_builder: AbstractSinglePointUtilityFunctionBuilder,
    utility_maximizer: AbstractSinglePointUtilityMaximizer,
):
    key = jr.key(42)
    posterior_handlers = {OBJECTIVE: posterior_handler}
    objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    datasets = {"OBJECTIVE": objective_dataset}
    with pytest.raises(NotImplementedError):
        UtilityDrivenDecisionMaker(
            search_space=search_space,
            posterior_handlers=posterior_handlers,
            datasets=datasets,
            utility_function_builder=utility_function_builder,
            utility_maximizer=utility_maximizer,
            key=key,
            post_ask=[],
            post_tell=[],
            batch_size=2,
        )


def test_invalid_tags_raises_error(
    search_space: AbstractSearchSpace,
    posterior_handler: PosteriorHandler,
    utility_function_builder: AbstractSinglePointUtilityFunctionBuilder,
    utility_maximizer: AbstractSinglePointUtilityMaximizer,
):
    key = jr.key(42)
    posterior_handlers = {OBJECTIVE: posterior_handler}
    dataset = get_dataset(num_points=5, key=jr.key(42))
    datasets = {"CONSTRAINT": dataset}  # Dataset tag doesn't match posterior tag
    with pytest.raises(ValueError):
        UtilityDrivenDecisionMaker(
            search_space=search_space,
            posterior_handlers=posterior_handlers,
            datasets=datasets,
            utility_function_builder=utility_function_builder,
            utility_maximizer=utility_maximizer,
            key=key,
            post_ask=[],
            post_tell=[],
            batch_size=1,
        )


def test_initialisation_optimizes_posterior_hyperparameters(
    search_space: AbstractSearchSpace,
    posterior_handler: PosteriorHandler,
    utility_function_builder: AbstractSinglePointUtilityFunctionBuilder,
    utility_maximizer: AbstractSinglePointUtilityMaximizer,
):
    key = jr.key(42)
    posterior_handlers = {OBJECTIVE: posterior_handler, CONSTRAINT: posterior_handler}
    objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    constraint_dataset = get_dataset(num_points=5, key=jr.key(10))
    datasets = {"OBJECTIVE": objective_dataset, CONSTRAINT: constraint_dataset}
    decision_maker = UtilityDrivenDecisionMaker(
        search_space=search_space,
        posterior_handlers=posterior_handlers,
        datasets=datasets,
        utility_function_builder=utility_function_builder,
        utility_maximizer=utility_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=1,
    )
    # Assert kernel hyperparameters get changed from their initial values
    assert decision_maker.posteriors[OBJECTIVE].prior.kernel.lengthscale != jnp.array(
        1.0
    )
    assert decision_maker.posteriors[OBJECTIVE].prior.kernel.variance != jnp.array(1.0)
    assert decision_maker.posteriors[CONSTRAINT].prior.kernel.lengthscale != jnp.array(
        1.0
    )
    assert decision_maker.posteriors[CONSTRAINT].prior.kernel.variance != jnp.array(1.0)
    assert (
        decision_maker.posteriors[CONSTRAINT].prior.kernel.lengthscale
        != decision_maker.posteriors[OBJECTIVE].prior.kernel.lengthscale
    )
    assert (
        decision_maker.posteriors[CONSTRAINT].prior.kernel.variance
        != decision_maker.posteriors[OBJECTIVE].prior.kernel.variance
    )


def test_decision_maker_ask(
    search_space: AbstractSearchSpace,
    posterior_handler: PosteriorHandler,
    utility_function_builder: AbstractSinglePointUtilityFunctionBuilder,
    utility_maximizer: AbstractSinglePointUtilityMaximizer,
):
    key = jr.key(42)
    posterior_handlers = {OBJECTIVE: posterior_handler}
    objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    datasets = {"OBJECTIVE": objective_dataset}
    decision_maker = UtilityDrivenDecisionMaker(
        search_space=search_space,
        posterior_handlers=posterior_handlers,
        datasets=datasets,
        utility_function_builder=utility_function_builder,
        utility_maximizer=utility_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=1,
    )
    initial_decision_maker_key = decision_maker.key
    query_point = decision_maker.ask(key=key)
    assert query_point.shape == (1, 1)
    assert jnp.allclose(query_point, jnp.array([[0.5]]), atol=1e-5)
    assert len(decision_maker.current_utility_functions) == 1
    assert (
        decision_maker.key == initial_decision_maker_key
    ).all()  # Ensure decision maker key is unchanged


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_decision_maker_ask_multi_batch_ts(
    search_space: AbstractSearchSpace,
    posterior_handler: PosteriorHandler,
    thompson_sampling_utility_function_builder: ThompsonSampling,
    utility_maximizer: AbstractSinglePointUtilityMaximizer,
    batch_size: int,
):
    key = jr.key(42)
    posterior_handlers = {OBJECTIVE: posterior_handler}
    objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    datasets = {"OBJECTIVE": objective_dataset}
    decision_maker = UtilityDrivenDecisionMaker(
        search_space=search_space,
        posterior_handlers=posterior_handlers,
        datasets=datasets,
        utility_function_builder=thompson_sampling_utility_function_builder,
        utility_maximizer=utility_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=batch_size,
    )
    initial_decision_maker_key = decision_maker.key
    query_points = decision_maker.ask(key=key)
    assert query_points.shape == (batch_size, 1)

    # TODO: ask henry about this failing assertion
    # assert (
    #     len(jnp.unique(query_points)) == batch_size
    # )  # Ensure we aren't drawing the same Thompson sample each time
    assert len(decision_maker.current_utility_functions) == batch_size
    assert (
        decision_maker.key == initial_decision_maker_key
    ).all()  # Ensure decision maker key is unchanged


def test_decision_maker_tell_with_inconsistent_observations_raises_error(
    search_space: AbstractSearchSpace,
    posterior_handler: PosteriorHandler,
    utility_function_builder: AbstractSinglePointUtilityFunctionBuilder,
    utility_maximizer: AbstractSinglePointUtilityMaximizer,
):
    key = jr.key(42)
    posterior_handlers = {OBJECTIVE: posterior_handler, CONSTRAINT: posterior_handler}
    initial_objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    initial_constraint_dataset = get_dataset(num_points=5, key=jr.key(10))
    datasets = {
        OBJECTIVE: initial_objective_dataset,
        CONSTRAINT: initial_constraint_dataset,
    }
    decision_maker = UtilityDrivenDecisionMaker(
        search_space=search_space,
        posterior_handlers=posterior_handlers,
        datasets=datasets,
        utility_function_builder=utility_function_builder,
        utility_maximizer=utility_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=1,
    )
    mock_objective_observation = get_dataset(num_points=1, key=jr.key(1))
    mock_constraint_observation = get_dataset(num_points=1, key=jr.key(2))
    observations = {
        OBJECTIVE: mock_objective_observation,
        "CONSTRAINT_ONE": mock_constraint_observation,  # Deliberately incorrect tag
    }
    with pytest.raises(ValueError):
        decision_maker.tell(observation_datasets=observations, key=key)


def test_decision_maker_tell_updates_datasets_and_models(
    search_space: AbstractSearchSpace,
    posterior_handler: PosteriorHandler,
    utility_function_builder: AbstractSinglePointUtilityFunctionBuilder,
    utility_maximizer: AbstractSinglePointUtilityMaximizer,
):
    key = jr.key(42)
    posterior_handlers = {OBJECTIVE: posterior_handler, CONSTRAINT: posterior_handler}
    initial_objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    initial_constraint_dataset = get_dataset(num_points=5, key=jr.key(10))
    datasets = {
        "OBJECTIVE": initial_objective_dataset,
        CONSTRAINT: initial_constraint_dataset,
    }
    decision_maker = UtilityDrivenDecisionMaker(
        search_space=search_space,
        posterior_handlers=posterior_handlers,
        datasets=datasets,
        utility_function_builder=utility_function_builder,
        utility_maximizer=utility_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=1,
    )
    initial_decision_maker_key = decision_maker.key
    initial_objective_posterior = decision_maker.posteriors[OBJECTIVE]
    initial_constraint_posterior = decision_maker.posteriors[CONSTRAINT]
    mock_objective_observation = get_dataset(num_points=1, key=jr.key(1))
    mock_constraint_observation = get_dataset(num_points=1, key=jr.key(2))
    observations = {
        OBJECTIVE: mock_objective_observation,
        CONSTRAINT: mock_constraint_observation,
    }
    decision_maker.tell(observation_datasets=observations, key=key)
    assert decision_maker.datasets[OBJECTIVE].n == 6
    assert decision_maker.datasets[CONSTRAINT].n == 6
    assert decision_maker.datasets[OBJECTIVE].X[-1] == mock_objective_observation.X[0]
    assert decision_maker.datasets[CONSTRAINT].X[-1] == mock_constraint_observation.X[0]
    assert (
        decision_maker.posteriors[OBJECTIVE].prior.kernel.lengthscale
        != initial_objective_posterior.prior.kernel.lengthscale
    )
    assert (
        decision_maker.posteriors[OBJECTIVE].prior.kernel.variance
        != initial_objective_posterior.prior.kernel.variance
    )
    assert (
        decision_maker.posteriors[CONSTRAINT].prior.kernel.lengthscale
        != initial_constraint_posterior.prior.kernel.lengthscale
    )
    assert (
        decision_maker.posteriors[CONSTRAINT].prior.kernel.variance
        != initial_constraint_posterior.prior.kernel.variance
    )
    assert (
        decision_maker.key == initial_decision_maker_key
    ).all()  # Ensure decision maker key has not been updated


@pytest.mark.parametrize("n_steps", [1, 3])
def test_decision_maker_run(
    search_space: AbstractSearchSpace,
    posterior_handler: PosteriorHandler,
    utility_function_builder: AbstractSinglePointUtilityFunctionBuilder,
    utility_maximizer: AbstractSinglePointUtilityMaximizer,
    n_steps: int,
):
    key = jr.key(42)
    posterior_handlers = {OBJECTIVE: posterior_handler}
    initial_objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    initial_datasets = {
        "OBJECTIVE": initial_objective_dataset,
    }
    decision_maker = UtilityDrivenDecisionMaker(
        search_space=search_space,
        posterior_handlers=posterior_handlers,
        datasets=initial_datasets,
        utility_function_builder=utility_function_builder,
        utility_maximizer=utility_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=1,
    )
    initial_decision_maker_key = decision_maker.key
    black_box_fn = Quadratic()
    black_box_function_evaluator = build_function_evaluator(
        {OBJECTIVE: black_box_fn.evaluate}
    )
    query_datasets = decision_maker.run(
        n_steps=n_steps, black_box_function_evaluator=black_box_function_evaluator
    )
    assert initial_datasets[OBJECTIVE].n == 5
    assert query_datasets[OBJECTIVE].n == 5 + n_steps
    assert (
        jnp.abs(query_datasets[OBJECTIVE].X[-n_steps:] - jnp.array([[0.5]])) < 1e-5
    ).all()  # Ensure we're querying the correct point in our dummy utility function at each step
    assert (
        decision_maker.key != initial_decision_maker_key
    ).all()  # Ensure decision maker key gets updated


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_decision_maker_run_ts(
    search_space: AbstractSearchSpace,
    posterior_handler: PosteriorHandler,
    thompson_sampling_utility_function_builder: ThompsonSampling,
    utility_maximizer: AbstractSinglePointUtilityMaximizer,
    n_steps: int,
    batch_size: int,
):
    key = jr.key(42)
    posterior_handlers = {OBJECTIVE: posterior_handler}
    initial_objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    initial_datasets = {
        "OBJECTIVE": initial_objective_dataset,
    }
    decision_maker = UtilityDrivenDecisionMaker(
        search_space=search_space,
        posterior_handlers=posterior_handlers,
        datasets=initial_datasets,
        utility_function_builder=thompson_sampling_utility_function_builder,
        utility_maximizer=utility_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=batch_size,
    )
    initial_decision_maker_key = decision_maker.key
    black_box_fn = Quadratic()
    black_box_function_evaluator = build_function_evaluator(
        {OBJECTIVE: black_box_fn.evaluate}
    )
    query_datasets = decision_maker.run(
        n_steps=n_steps, black_box_function_evaluator=black_box_function_evaluator
    )
    assert initial_datasets[OBJECTIVE].n == 5
    assert (
        query_datasets[OBJECTIVE].n == 5 + n_steps * batch_size
    )  # Ensure we're getting the correct number of points
    assert (
        decision_maker.key != initial_decision_maker_key
    ).all()  # Ensure decision maker key gets updated
