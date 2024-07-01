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

from beartype.typing import Mapping
from jaxtyping import Num

from gpjax.dataset import Dataset
from gpjax.decision_making.utility_functions.base import (
    AbstractSinglePointUtilityFunctionBuilder,
    SinglePointUtilityFunction,
)
from gpjax.decision_making.utils import (
    OBJECTIVE,
    gaussian_cdf,
)
from gpjax.gps import ConjugatePosterior
from gpjax.typing import (
    Array,
    KeyArray,
)


@dataclass
class ProbabilityOfImprovement(AbstractSinglePointUtilityFunctionBuilder):
    """
    TODO: write.
    """

    def build_utility_function(
        self,
        posteriors: Mapping[str, ConjugatePosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> SinglePointUtilityFunction:
        """
        Draw an approximate sample from the posterior of the objective model and return
        the *negative* of this sample as a utility function, as utility functions
        are *maximised*.

        Args:
            posteriors (Mapping[str, AbstractPosterior]): Dictionary of posteriors to be
            used to form the utility function. One of the posteriors must correspond
            to the `OBJECTIVE` key, as we sample from the objective posterior to form
            the utility function.
            datasets (Mapping[str, Dataset]): Dictionary of datasets which may be used
            to form the utility function. Keys in `datasets` should correspond to
            keys in `posteriors`. One of the datasets must correspond
            to the `OBJECTIVE` key.
            key (KeyArray): JAX PRNG key used for random number generation. This can be
            changed to draw different samples.

        Returns:
            SinglePointUtilityFunction: An appproximate sample from the objective model
            posterior to to be *maximised* in order to decide which point to query
            next.
        """
        self.check_objective_present(posteriors, datasets)

        objective_posterior = posteriors[OBJECTIVE]
        if not isinstance(objective_posterior, ConjugatePosterior):
            raise ValueError(
                "Objective posterior must be a ConjugatePosterior to draw an approximate sample."
            )

        objective_dataset = datasets[OBJECTIVE]

        def probability_of_improvement(x_test: Num[Array, "N D"]):
            predictive_dist = objective_posterior.predict(x_test, objective_dataset)

            # Assuming that the goal is to minimize the objective function
            best_y = objective_dataset.y.min()

            return gaussian_cdf(
                (best_y - predictive_dist.mean()) / predictive_dist.stddev()
            ).reshape(-1, 1)

        return probability_of_improvement  # Utility functions are *maximised*
