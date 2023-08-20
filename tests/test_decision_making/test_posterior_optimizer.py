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
from jax.config import config

config.update("jax_enable_x64", True)

from beartype.typing import (
    Callable,
    Union,
)
import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax.decision_making.posterior_optimizer import (
    AbstractPosteriorOptimizer,
    AdamPosteriorOptimizer,
)
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
    AbstractObjective,
    ConjugateMLL,
    NonConjugateMLL,
)


def gaussian_likelihood_builder(num_datapoints: int) -> Gaussian:
    return Gaussian(num_datapoints=num_datapoints)


def poisson_likelihood_builder(num_datapoints: int) -> Poisson:
    return Poisson(num_datapoints=num_datapoints)


def test_abstract_posterior_optimizer():
    with pytest.raises(TypeError):
        AbstractPosteriorOptimizer()


@pytest.mark.parametrize("num_iters", [0, -1, -10])
def test_adam_posterior_optimizer_erroneous_num_iters_raises_error(num_iters: int):
    with pytest.raises(ValueError):
        objective = ConjugateMLL(negative=True)
        AdamPosteriorOptimizer(
            num_iters=num_iters, learning_rate=0.01, objective=objective
        )


@pytest.mark.parametrize("learning_rate", [0.0, -1.0, -10.0])
def test_adam_posterior_optimizer_erroneous_learning_rate_raises_error(
    learning_rate: float,
):
    with pytest.raises(ValueError):
        objective = ConjugateMLL(negative=True)
        AdamPosteriorOptimizer(
            num_iters=10, learning_rate=learning_rate, objective=objective
        )


@pytest.mark.parametrize("num_datapoints", [5, 50])
@pytest.mark.parametrize(
    "likelihood_builder, training_objective, test_function",
    [
        (gaussian_likelihood_builder, ConjugateMLL(negative=True), Forrester()),
        (
            poisson_likelihood_builder,
            NonConjugateMLL(negative=True),
            PoissonTestFunction(),
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_adam_posterior_optimize(
    num_datapoints: int,
    likelihood_builder: Callable[[int], AbstractLikelihood],
    training_objective: AbstractObjective,
    test_function: Union[Forrester, PoissonTestFunction],
):
    mean_function = Constant(constant=jnp.array([1.0]))
    kernel = Matern52(lengthscale=jnp.array([0.5]), variance=jnp.array(1.0))
    prior = Prior(mean_function=mean_function, kernel=kernel)
    posterior_optimizer = AdamPosteriorOptimizer(
        num_iters=10, learning_rate=0.01, objective=training_objective
    )
    dataset = test_function.generate_dataset(
        num_points=num_datapoints, key=jr.PRNGKey(42)
    )
    initial_posterior = prior * likelihood_builder(num_datapoints)
    optimized_posterior = posterior_optimizer.optimize(
        posterior=initial_posterior, dataset=dataset, key=jr.PRNGKey(42)
    )
    assert type(optimized_posterior) == type(initial_posterior)
    assert training_objective(optimized_posterior, dataset) < training_objective(
        initial_posterior, dataset
    )  # Ensure optimization reduces training objective
