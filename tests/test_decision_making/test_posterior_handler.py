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

from beartype.typing import (
    Callable,
    Union,
)
import jax.numpy as jnp
import jax.random as jr
import optax as ox
import pytest

from gpjax.decision_making.posterior_handler import PosteriorHandler
from gpjax.decision_making.test_functions import (
    Forrester,
    PoissonTestFunction,
)
from gpjax.gps import Prior
from gpjax.kernels import Matern52
from gpjax.likelihoods import (
    AbstractLikelihood,
    Gaussian,
    Poisson,
)
from gpjax.mean_functions import Constant
from gpjax.objectives import (
    Objective,
    conjugate_mll,
    non_conjugate_mll,
)


def gaussian_likelihood_builder(num_datapoints: int) -> Gaussian:
    return Gaussian(num_datapoints=num_datapoints)


def poisson_likelihood_builder(num_datapoints: int) -> Poisson:
    return Poisson(num_datapoints=num_datapoints)


@pytest.mark.parametrize("num_optimization_iters", [0, -1, -10])
def test_posterior_handler_erroneous_num_optimization_iterations_raises_error(
    num_optimization_iters: int,
):
    mean_function = Constant()
    kernel = Matern52()
    prior = Prior(mean_function=mean_function, kernel=kernel)
    likelihood_builder = gaussian_likelihood_builder
    with pytest.raises(ValueError):
        PosteriorHandler(
            prior=prior,
            likelihood_builder=likelihood_builder,
            optimization_objective=conjugate_mll,
            optimizer=ox.adam(learning_rate=0.01),
            num_optimization_iters=num_optimization_iters,
        )


def test_get_optimized_posterior_with_no_key_raises_error():
    mean_function = Constant()
    kernel = Matern52()
    prior = Prior(mean_function=mean_function, kernel=kernel)
    likelihood_builder = gaussian_likelihood_builder
    posterior_handler = PosteriorHandler(
        prior=prior,
        likelihood_builder=likelihood_builder,
        optimization_objective=conjugate_mll,
        optimizer=ox.adam(learning_rate=0.01),
        num_optimization_iters=10,
    )
    toy_function = Forrester()
    dataset = toy_function.generate_dataset(num_points=5, key=jr.key(42))
    with pytest.raises(ValueError):
        posterior_handler.get_posterior(dataset=dataset, optimize=True)


def test_update_and_optimize_posterior_with_no_key_raises_error():
    mean_function = Constant()
    kernel = Matern52()
    prior = Prior(mean_function=mean_function, kernel=kernel)
    likelihood_builder = gaussian_likelihood_builder
    posterior_handler = PosteriorHandler(
        prior=prior,
        likelihood_builder=likelihood_builder,
        optimization_objective=conjugate_mll,
        optimizer=ox.adam(learning_rate=0.01),
        num_optimization_iters=10,
    )
    toy_function = Forrester()
    dataset = toy_function.generate_dataset(num_points=5, key=jr.key(42))
    initial_posterior = posterior_handler.get_posterior(dataset=dataset, optimize=False)
    with pytest.raises(ValueError):
        posterior_handler.update_posterior(
            dataset=dataset, previous_posterior=initial_posterior, optimize=True
        )


@pytest.mark.parametrize("num_datapoints", [1, 50])
@pytest.mark.parametrize(
    "likelihood_builder, training_objective, test_function",
    [
        (gaussian_likelihood_builder, conjugate_mll, Forrester()),
        (
            poisson_likelihood_builder,
            non_conjugate_mll,
            PoissonTestFunction(),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_get_posterior_no_optimization_correct_num_datapoints_and_not_optimized(
    num_datapoints: int,
    likelihood_builder: Callable[[int], AbstractLikelihood],
    training_objective: Objective,
    test_function: Union[Forrester, PoissonTestFunction],
):
    mean_function = Constant(constant=jnp.array([1.0]))
    kernel = Matern52(lengthscale=jnp.array([0.5]), variance=jnp.array(1.0))
    prior = Prior(mean_function=mean_function, kernel=kernel)
    posterior_handler = PosteriorHandler(
        prior=prior,
        likelihood_builder=likelihood_builder,
        optimization_objective=training_objective,
        optimizer=ox.adam(learning_rate=0.01),
        num_optimization_iters=10,
    )
    dataset = test_function.generate_dataset(num_points=num_datapoints, key=jr.key(42))
    posterior = posterior_handler.get_posterior(dataset=dataset, optimize=False)
    assert posterior.likelihood.num_datapoints == num_datapoints
    assert posterior.prior.mean_function.constant.value == jnp.array([1.0])
    assert posterior.prior.kernel.lengthscale.value == jnp.array([0.5])
    assert posterior.prior.kernel.variance.value == jnp.array(1.0)


@pytest.mark.parametrize("num_datapoints", [5, 50])
@pytest.mark.parametrize(
    "likelihood_builder, training_objective, test_function",
    [
        (gaussian_likelihood_builder, conjugate_mll, Forrester()),
        (
            poisson_likelihood_builder,
            non_conjugate_mll,
            PoissonTestFunction(),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_get_posterior_with_optimization_correct_num_datapoints_and_optimized(
    num_datapoints: int,
    likelihood_builder: Callable[[int], AbstractLikelihood],
    training_objective: Objective,
    test_function: Union[Forrester, PoissonTestFunction],
):
    mean_function = Constant(constant=jnp.array([1.0]))
    kernel = Matern52(lengthscale=jnp.array([0.5]), variance=jnp.array(1.0))
    prior = Prior(mean_function=mean_function, kernel=kernel)
    non_optimized_posterior = prior * likelihood_builder(num_datapoints)
    posterior_handler = PosteriorHandler(
        prior=prior,
        likelihood_builder=likelihood_builder,
        optimization_objective=training_objective,
        optimizer=ox.adam(learning_rate=0.01),
        num_optimization_iters=10,
    )
    dataset = test_function.generate_dataset(num_points=num_datapoints, key=jr.key(42))
    optimized_posterior = posterior_handler.get_posterior(
        dataset=dataset, optimize=True, key=jr.key(42)
    )
    assert optimized_posterior.likelihood.num_datapoints == num_datapoints
    assert optimized_posterior.prior.mean_function.constant != jnp.array([1.0])
    assert optimized_posterior.prior.kernel.lengthscale != jnp.array([0.5])
    assert optimized_posterior.prior.kernel.variance != jnp.array(1.0)
    assert training_objective(optimized_posterior, dataset) < training_objective(
        non_optimized_posterior, dataset
    )  # Ensure optimization reduces training objective


@pytest.mark.parametrize("initial_num_datapoints", [5, 50])
@pytest.mark.parametrize(
    "likelihood_builder, training_objective, test_function",
    [
        (gaussian_likelihood_builder, conjugate_mll, Forrester()),
        (
            poisson_likelihood_builder,
            non_conjugate_mll,
            PoissonTestFunction(),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_update_posterior_no_optimize_same_prior_parameters_and_different_num_datapoints(
    initial_num_datapoints: int,
    likelihood_builder: Callable[[int], AbstractLikelihood],
    training_objective: Objective,
    test_function: Union[Forrester, PoissonTestFunction],
):
    mean_function = Constant(constant=jnp.array([1.0]))
    kernel = Matern52(lengthscale=jnp.array([0.5]), variance=jnp.array(1.0))
    prior = Prior(mean_function=mean_function, kernel=kernel)
    posterior_handler = PosteriorHandler(
        prior=prior,
        likelihood_builder=likelihood_builder,
        optimization_objective=training_objective,
        optimizer=ox.adam(learning_rate=0.01),
        num_optimization_iters=10,
    )
    initial_dataset = test_function.generate_dataset(
        num_points=initial_num_datapoints, key=jr.key(42)
    )
    initial_posterior = posterior_handler.get_posterior(
        dataset=initial_dataset, optimize=False
    )
    updated_dataset = initial_dataset + test_function.generate_dataset(
        num_points=1, key=jr.key(42)
    )
    assert updated_dataset.n == initial_dataset.n + 1
    updated_posterior = posterior_handler.update_posterior(
        dataset=updated_dataset, previous_posterior=initial_posterior, optimize=False
    )
    assert (
        updated_posterior.prior.kernel.lengthscale
        == initial_posterior.prior.kernel.lengthscale
    )
    assert (
        updated_posterior.prior.kernel.variance
        == initial_posterior.prior.kernel.variance
    )
    assert (
        updated_posterior.prior.mean_function.constant
        == initial_posterior.prior.mean_function.constant
    )
    assert updated_posterior.likelihood.num_datapoints == updated_dataset.n


@pytest.mark.parametrize("initial_num_datapoints", [5, 50])
@pytest.mark.parametrize(
    "likelihood_builder, training_objective, test_function",
    [
        (gaussian_likelihood_builder, conjugate_mll, Forrester()),
        (
            poisson_likelihood_builder,
            non_conjugate_mll,
            PoissonTestFunction(),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_update_posterior_with_optimization_updated_prior_parameters_and_different_num_datapoints(
    initial_num_datapoints: int,
    likelihood_builder: Callable[[int], AbstractLikelihood],
    training_objective: Objective,
    test_function: Union[Forrester, PoissonTestFunction],
):
    mean_function = Constant(constant=jnp.array([1.0]))
    kernel = Matern52(lengthscale=jnp.array([0.5]), variance=jnp.array(1.0))
    prior = Prior(mean_function=mean_function, kernel=kernel)
    posterior_handler = PosteriorHandler(
        prior=prior,
        likelihood_builder=likelihood_builder,
        optimization_objective=training_objective,
        optimizer=ox.adam(learning_rate=0.01),
        num_optimization_iters=10,
    )
    initial_dataset = test_function.generate_dataset(
        num_points=initial_num_datapoints, key=jr.key(42)
    )
    initial_posterior = posterior_handler.get_posterior(
        dataset=initial_dataset, optimize=False
    )
    updated_dataset = initial_dataset + test_function.generate_dataset(
        num_points=1, key=jr.key(42)
    )
    assert updated_dataset.n == initial_dataset.n + 1
    non_optimized_updated_posterior = posterior_handler.update_posterior(
        dataset=updated_dataset, previous_posterior=initial_posterior, optimize=False
    )
    optimized_updated_posterior = posterior_handler.update_posterior(
        dataset=updated_dataset,
        previous_posterior=initial_posterior,
        optimize=True,
        key=jr.key(42),
    )
    assert (
        optimized_updated_posterior.prior.kernel.lengthscale
        != initial_posterior.prior.kernel.lengthscale
    )
    assert (
        optimized_updated_posterior.prior.kernel.variance
        != initial_posterior.prior.kernel.variance
    )
    assert (
        optimized_updated_posterior.prior.mean_function.constant
        != initial_posterior.prior.mean_function.constant
    )
    assert optimized_updated_posterior.likelihood.num_datapoints == updated_dataset.n
    assert training_objective(
        optimized_updated_posterior, updated_dataset
    ) < training_objective(
        non_optimized_updated_posterior, updated_dataset
    )  # Ensure optimization reduces training objective
