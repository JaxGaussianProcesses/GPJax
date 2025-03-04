# Copyright 2024 The JaxGaussianProcesses Contributors. All Rights Reserved.
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

from gpjax.decision_making.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    Forrester,
    LogarithmicGoldsteinPrice,
)
from gpjax.decision_making.utility_functions.expected_improvement import (
    ExpectedImprovement,
)
from gpjax.decision_making.utils import (
    OBJECTIVE,
    get_best_latent_observation_val,
)
from gpjax.typing import KeyArray
from tests.test_decision_making.utils import generate_dummy_conjugate_posterior


@pytest.mark.parametrize(
    "test_target_function",
    [Forrester(), LogarithmicGoldsteinPrice()],
)
@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10)])
def test_expected_improvement_utility_function_correct_values(
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    # Test validity of computed values with Monte-Carlo
    dataset = test_target_function.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    ei_fn = ExpectedImprovement().build_utility_function(posteriors, datasets, key)
    test_x = test_target_function.generate_test_points(100, key)
    ei = ei_fn(test_x)
    latent_dist = posterior.predict(test_x, dataset)
    pred_dist = posterior.likelihood(latent_dist)
    pred_mean = pred_dist.mean()
    pred_var = pred_dist.variance()
    samples = tfp.distributions.Normal(loc=pred_mean, scale=jnp.sqrt(pred_var)).sample(
        1000, seed=key
    )
    eta = get_best_latent_observation_val(posterior, dataset)
    mc_ei = jnp.expand_dims(jnp.mean(jnp.maximum(eta - samples, 0), 0), -1)
    assert jnp.all(ei >= 0)
    assert jnp.allclose(ei, mc_ei, rtol=0.08, atol=1e-6)
