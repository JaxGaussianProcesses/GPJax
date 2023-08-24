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

from beartype.typing import (
    Callable,
    Optional,
)
import optax as ox

import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.gps import (
    AbstractLikelihood,
    AbstractPosterior,
    AbstractPrior,
)
from gpjax.objectives import AbstractObjective
from gpjax.typing import KeyArray

LikelihoodBuilder = Callable[[int], AbstractLikelihood]
"""Type alias for likelihood builders, which take the number of datapoints as input and
return a likelihood object initialised with the given number of datapoints."""


@dataclass
class PosteriorHandler:
    """
    Class for handling the creation and updating of a GP posterior as new data is
    observed.

    Attributes:
        prior (AbstractPrior): Prior to use when forming the posterior.
        likelihood_builder (LikelihoodBuilder): Function which takes the number of
        datapoints as input and returns a likelihood object initialised with the given
        number of datapoints.
        optimization_objective (AbstractObjective): Objective to use for optimizing the
        posterior hyperparameters.
        optimizer (ox.GradientTransformation): Optax optimizer to use for optimizing the
        posterior hyperparameters.
        num_optimization_iterations (int): Number of iterations to optimize
        the posterior hyperparameters for.
    """

    prior: AbstractPrior
    likelihood_builder: LikelihoodBuilder
    optimization_objective: AbstractObjective
    optimizer: ox.GradientTransformation
    num_optimization_iters: int

    def __post_init__(self):
        if self.num_optimization_iters < 1:
            raise ValueError("num_optimization_iters must be greater than 0.")

    def get_posterior(
        self, dataset: Dataset, optimize: bool, key: Optional[KeyArray] = None
    ) -> AbstractPosterior:
        """
        Initialise (and optionally optimize) a posterior using the given dataset.

        Args:
            dataset (Dataset): Dataset to get posterior for.
            optimize (bool): Whether to optimize the posterior hyperparameters.
            key (Optional[KeyArray]): A JAX PRNG key which is used for optimizing the posterior
            hyperparameters.

        Returns:
            Posterior for the given dataset.
        """
        posterior = self.prior * self.likelihood_builder(dataset.n)

        if optimize:
            if key is None:
                raise ValueError(
                    "A key must be provided in order to optimize the posterior."
                )
            posterior = self._optimize_posterior(posterior, dataset, key)

        return posterior

    def update_posterior(
        self,
        dataset: Dataset,
        previous_posterior: AbstractPosterior,
        optimize: bool,
        key: Optional[KeyArray] = None,
    ) -> AbstractPosterior:
        """
        Update the given posterior with the given dataset. This needs to be done when
        the number of datapoints in the (training) dataset of the posterior changes, as
        the `AbstractLikelihood` class requires the number of datapoints to be specified.
        Hyperparameters may or may not be optimized, depending on the value of the
        `optimize` parameter. Note that the updated poterior will be initialised with
        the same prior hyperparameters as the previous posterior, but the likelihood
        will be re-initialised with the new number of datapoints, and hyperparameters
        set as in the `likelihood_builder` function.

        Args:
            dataset: Dataset to get posterior for.
            previous_posterior: Posterior being updated. This is supplied as one may
            wish to simply increase the number of datapoints in the likelihood, without
            optimizing the posterior hyperparameters, in which case the previous
            posterior can be used to obtain the previously set prior hyperparameters.
            optimize: Whether to optimize the posterior hyperparameters.
            key: A JAX PRNG key which is used for optimizing the posterior
            hyperparameters.
        """
        posterior = previous_posterior.prior * self.likelihood_builder(dataset.n)

        if optimize:
            if key is None:
                raise ValueError(
                    "A key must be provided in order to optimize the posterior."
                )
            posterior = self._optimize_posterior(posterior, dataset, key)
        return posterior

    def _optimize_posterior(
        self, posterior: AbstractPosterior, dataset: Dataset, key: KeyArray
    ) -> AbstractPosterior:
        """
        Takes a posterior and corresponding dataset and optimizes the posterior using the
        GPJax `fit` method.

        Args:
            posterior: Posterior being optimized.
            dataset: Dataset used for optimizing posterior.
            key: A JAX PRNG key for generating random numbers.
        Returns:
            Optimized posterior.
        """
        opt_posterior, _ = gpx.fit(
            model=posterior,
            objective=self.optimization_objective,
            train_data=dataset,
            optim=self.optimizer,
            num_iters=self.num_optimization_iters,
            safe=True,
            key=key,
            verbose=False,
        )

        return opt_posterior
