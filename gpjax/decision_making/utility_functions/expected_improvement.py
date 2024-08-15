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
from dataclasses import dataclass
from functools import partial

from beartype.typing import Mapping
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from gpjax.dataset import Dataset
from gpjax.decision_making.utility_functions.base import (
    AbstractSinglePointUtilityFunctionBuilder,
    SinglePointUtilityFunction,
)
from gpjax.decision_making.utils import (
    OBJECTIVE,
    get_best_latent_observation_val,
)
from gpjax.gps import ConjugatePosterior
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
)


@dataclass
class ExpectedImprovement(AbstractSinglePointUtilityFunctionBuilder):
    """
    Expected Improvement acquisition function as introduced by [Močkus,
    1974](https://link.springer.com/chapter/10.1007/3-540-07165-2_55). The "best"
    incumbent value is defined as the lowest posterior mean value evaluated at the the
    previously observed points. This enables the acquisition function to be utilised with noisy observations.
    """

    def build_utility_function(
        self,
        posteriors: Mapping[str, ConjugatePosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> SinglePointUtilityFunction:
        r"""
        Build the Expected Improvement acquisition function. This computes the expected
        improvement over the "best" of the previously observed points, utilising the
        posterior distribution of the surrogate model. For posterior distribution
        $`f(\cdot)`$, and best incumbent value $`\eta`$, this is defined
        as:
        ```math
        \alpha_{\text{EI}}(\mathbf{x}) = \mathbb{E}\left[\max(0, \eta - f(\mathbf{x}))\right]
        ```

        Args:
            posteriors (Mapping[str, ConjugatePosterior]): Dictionary of posteriors to
            used to form the utility function. One posteriors must correspond to the
            `OBJECTIVE` key, as we utilise the objective posterior to form the utility
            function.
            datasets (Mapping[str, Dataset]): Dictionary of datasets used to form the
            utility function. Keys in `datasets` should correspond to keys in
            `posteriors`. One of the datasets must correspond to the `OBJECTIVE` key.
            key (KeyArray): JAX PRNG key used for random number generation.

        Returns:
            SinglePointUtilityFunction: The Expected Improvement acquisition function to
            to be *maximised* in order to decide which point to query next.
        """
        self.check_objective_present(posteriors, datasets)
        objective_posterior = posteriors[OBJECTIVE]
        objective_dataset = datasets[OBJECTIVE]

        if not isinstance(objective_posterior, ConjugatePosterior):
            raise ValueError(
                "Objective posterior must be a ConjugatePosterior to compute the Expected Improvement."
            )

        if (
            objective_dataset.X is None
            or objective_dataset.n == 0
            or objective_dataset.y is None
        ):
            raise ValueError("Objective dataset must contain at least one item")

        eta = get_best_latent_observation_val(objective_posterior, objective_dataset)
        return partial(
            _expected_improvement, objective_posterior, objective_dataset, eta
        )


def _expected_improvement(
    objective_posterior: ConjugatePosterior,
    objective_dataset: Dataset,
    eta: Float[Array, ""],
    x: Float[Array, "N D"],
) -> Float[Array, "N 1"]:
    latent_dist = objective_posterior(x, objective_dataset)
    mean = latent_dist.mean()
    var = latent_dist.variance()
    normal = tfp.distributions.Normal(mean, jnp.sqrt(var))
    return jnp.expand_dims(
        ((eta - mean) * normal.cdf(eta) + var * normal.prob(eta)), -1
    )
