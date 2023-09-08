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
from dataclasses import dataclass

from beartype.typing import Mapping

from gpjax.dataset import Dataset
from gpjax.decision_making.utility_functions.base import (
    AbstractSinglePointUtilityFunctionBuilder,
    SinglePointUtilityFunction,
)
from gpjax.decision_making.utils import OBJECTIVE
from gpjax.gps import ConjugatePosterior
from gpjax.typing import KeyArray


@dataclass
class ThompsonSampling(AbstractSinglePointUtilityFunctionBuilder):
    """
    Form a utility function by drawing an approximate sample from the posterior,
    using decoupled sampling as introduced in [Wilson et. al.
    (2020)](https://arxiv.org/abs/2002.09309). Note that we return the *negative* of the
    sample as the utility function, as utility functions are *maximised*.

    Note that this is a single batch utility function, as it doesn't support classical
    batching. However, Thompson sampling can be used in a batched setting by drawing a
    batch of different samples from the GP posterior. This can be done by calling
    `build_utility_function` with different keys, an example of which can be seen in the
    `ask` method of the `UtilityDrivenDecisionMaker` class. The samples can then be
    optimised sequentially.

    Attributes:
        num_features (int): The number of random Fourier features to use when drawing
            the approximate sample from the posterior. Defaults to 100.
    """

    num_features: int = 100

    def __post_init__(self):
        if self.num_features <= 0:
            raise ValueError(
                "The number of random Fourier features must be a positive integer."
            )

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
        thompson_sample = objective_posterior.sample_approx(
            num_samples=1,
            train_data=objective_dataset,
            key=key,
            num_features=self.num_features,
        )

        return lambda x: -1.0 * thompson_sample(x)  # Utility functions are *maximised*
