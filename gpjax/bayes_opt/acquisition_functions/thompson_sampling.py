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
from typing import Mapping

from gpjax.bayes_opt.acquisition_functions.base import (
    AbstractAcquisitionFunctionBuilder,
    AcquisitionFunction,
)
from gpjax.bayes_opt.function_evaluator import OBJECTIVE
from gpjax.dataset import Dataset
from gpjax.gps import ConjugatePosterior
from gpjax.typing import KeyArray


@dataclass
class ThompsonSamplingAcquisitionFunctionBuilder(AbstractAcquisitionFunctionBuilder):
    """
    Form an acquisition function by drawing an approximate sample from the posterior,
    using decoupled sampling as introduced in [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309).
    """

    num_rff_features: int = 500

    def build_acquisition_function(
        self,
        posteriors: Mapping[str, ConjugatePosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> AcquisitionFunction:
        if OBJECTIVE not in posteriors.keys():
            raise ValueError("Objective posterior not found in posteriors")
        if OBJECTIVE not in datasets.keys():
            raise ValueError("Objective dataset not found in datasets")
        if key is None:
            raise ValueError(
                "Key must be provided to draw an approximate sample from the posterior"
            )

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
            num_features=self.num_rff_features,
        )

        return lambda x: -1.0 * thompson_sample(
            x
        )  # Acquisition functions are *maximised*
