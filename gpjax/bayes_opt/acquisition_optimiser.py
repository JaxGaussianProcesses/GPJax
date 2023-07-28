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

import jax.numpy as jnp
import jax.random as jr
from jaxopt import ScipyBoundedMinimize

from gpjax.bayes_opt.acquisition_functions import AcquisitionFunction
from gpjax.bayes_opt.search_space import (
    AbstractSearchSpace,
    BoxSearchSpace,
)
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
)


@dataclass
class AbstractAcquisitionOptimiser(ABC):
    """Abstract base class for acquisition function optimisers."""

    @abstractmethod
    def optimise(
        self,
        acquisition_function: AcquisitionFunction,
        search_space: AbstractSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        """Maximise the given acquisition function over the search space provided.

        Args:
            acquisition_function (AcquisitionFunction): Acquisition function to be
            optimised.
            search_space (AbstractSearchSpace): Search space over which to optimise
            the acquisition function.
            key (KeyArray): JAX PRNG key.

        Returns:
            Float[Array, "1 D"]: Point at which the acquisition function is maximised.
        """
        raise NotImplementedError


@dataclass
class ContinuousAcquisitionOptimiser(AbstractAcquisitionOptimiser):
    """The `ContinuousAcquisitionOptimiser` class is used to optimise acquisition
    functions over the continuous domain with L-BFGS-B. First we sample the acquisition
    function at `num_initial_samples` points uniformly at random from the search space,
    and then we run L-BFG-B from the best of these initial points."""

    num_initial_samples: int

    def optimise(
        self,
        acquisition_function: AcquisitionFunction,
        search_space: BoxSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        initial_sample_points = jr.uniform(
            key,
            shape=(self.num_initial_samples, search_space.dimensionality),
            minval=search_space.lower_bounds,
            maxval=search_space.upper_bounds,
        )  # [N, D] TODO: Specify float64?
        initial_sample_values = acquisition_function(initial_sample_points)  # [N, 1]
        print(
            f"Initial Sample Values Shape: {initial_sample_values.shape} Should be [N, 1]"
        )

        best_initial_sample_value_idx = jnp.argmax(
            initial_sample_values, axis=0, keepdims=True
        )
        best_initial_sample_point = jnp.take_along_axis(
            initial_sample_points, best_initial_sample_value_idx, axis=0
        )  # [1, D]
        print(
            f"Best Initial Sample Point Shape: {best_initial_sample_point.shape} Should be [1, D]"
        )

        # Jaxopt minimiser requires a function which returns a scalar. It calls the
        # acquisition function with one point at a time, so the acquisition function
        # returns an array of shape [1, 1], so  we index to return a scalar. Note that
        # we also return the negative of the acquisition function - this is because
        # acquisition functions should be *maximimised* but the Jaxopt minimiser
        # minimises functions.
        scalar_acquisition_fn = lambda x: -acquisition_function(x)[0][0]
        # TODO: Check shape of x being passed in above
        lbfgsb = ScipyBoundedMinimize(fun=scalar_acquisition_fn, method="l-bfgs-b")
        bounds = (search_space.lower_bounds, search_space.upper_bounds)
        optimised_point = lbfgsb.run(best_initial_sample_point, bounds=bounds).params
        print(f"Optimised Point Shape: {optimised_point.shape} Should be [1, D]")
        return optimised_point
