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

import optax as ox

import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.gps import AbstractPosterior
from gpjax.objectives import AbstractObjective
from gpjax.typing import KeyArray


@dataclass
class AbstractPosteriorOptimizer(ABC):
    """
    Abstract base class for optimizing a GP posterior.
    """

    @abstractmethod
    def optimize(
        self,
        posterior: AbstractPosterior,
        dataset: Dataset,
        key: KeyArray,
    ) -> AbstractPosterior:
        """
        Take a posterior and corresponding dataset and optimizes the posterior
        hyperparameters.

        Args:
            posterior: Posterior being optimized.
            dataset: Dataset used for optimizing posterior.
            key: A JAX PRNG key for generating random numbers.
        """
        raise NotImplementedError


@dataclass
class AdamPosteriorOptimizer(AbstractPosteriorOptimizer):
    """
    Class for optimizing posterior hyperparameters using Adam.

    Attributes:
        num_iters: The number of iterations of Adam to run.
        learning_rate: The learning rate for Adam.
        objective: The objective function to optimize.
    """

    num_iters: int
    learning_rate: float
    objective: AbstractObjective

    def __post_init__(self):
        if self.num_iters < 1:
            raise ValueError("num_iters must be greater than 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")

    def optimize(
        self,
        posterior: AbstractPosterior,
        dataset: Dataset,
        key: KeyArray,
    ) -> AbstractPosterior:
        """
        Take a posterior and corresponding dataset and optimizes the posterior using the
        Adam optimizer.

        Args:
            posterior: Posterior being optimized.
            dataset: Dataset used for optimizing posterior.
            key: A JAX PRNG key for generating random numbers.
        Returns:
            Optimized posterior.
        """
        opt_posterior, _ = gpx.fit(
            model=posterior,
            objective=self.objective,
            train_data=dataset,
            optim=ox.adam(learning_rate=self.learning_rate),
            num_iters=self.num_iters,
            safe=True,
            key=key,
            verbose=False,
        )

        return opt_posterior
