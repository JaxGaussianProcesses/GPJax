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
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

from beartype.typing import Mapping
from jax import jit
import optax as ox

import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.gps import AbstractPosterior
from gpjax.objectives import AbstractObjective
from gpjax.typing import KeyArray


@dataclass
class AbstractPosteriorOptimiser(ABC):
    """
    Abstract base class for optimising GP posteriors.
    """

    @abstractmethod
    def optimise(
        self,
        posteriors: Mapping[str, AbstractPosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> Mapping[str, AbstractPosterior]:
        """
        Take a set of posteriors and corresponding datasets and optimise the posterior
        hyperparameters.

        Args:
            posteriors: A mapping from tags to posteriors.
            datasets: A mapping from tags to datasets.
            key: A JAX PRNG key for generating random numbers.
        """
        raise NotImplementedError


@dataclass
class AdamPosteriorOptimiser(AbstractPosteriorOptimiser):
    """
    Class for optimising posterior hyperparameters using Adam.

    Attributes:
        num_iters: The number of iterations of Adam to run.
        learning_rate: The learning rate for Adam.
        objective: The objective function to optimise.
    """

    num_iters: int
    learning_rate: float
    objective: AbstractObjective

    def optimise(
        self,
        posteriors: Mapping[str, AbstractPosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> Mapping[str, AbstractPosterior]:
        opt_posteriors = {}
        for tag, posterior in posteriors.items():
            if tag not in datasets.keys():
                raise ValueError(
                    f"Trying to optimise model posterior with tag {tag} but no corresponding Dataset with tag {tag} found in datasets."
                )

            dataset = datasets[tag]
            objective = self.objective  # TODO: Test this with multiple models
            objective(posterior, datasets[tag])
            objective = jit(objective)
            print(f"Posterior: {posterior}")
            opt_posterior, _ = gpx.fit(
                model=posterior,
                objective=objective,
                train_data=dataset,
                optim=ox.adam(learning_rate=self.learning_rate),
                num_iters=self.num_iters,
                safe=True,
                key=key,
                verbose=False,
            )
            opt_posteriors[tag] = opt_posterior

        return opt_posteriors
