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

from gpjax.decision_making.search_space import (
    AbstractSearchSpace,
    ContinuousSearchSpace,
)
from gpjax.decision_making.utility_functions import SinglePointUtilityFunction
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
    ScalarFloat,
)


def _get_discrete_maximizer(
    query_points: Float[Array, "N D"], utility_function: SinglePointUtilityFunction
) -> Float[Array, "1 D"]:
    """Get the point which maximises the utility function evaluated at a given set of points.

    Args:
        query_points (Float[Array, "N D"]): Set of points at which to evaluate the
        utility function.
        utility_function (SinglePointUtilityFunction): Single point utility function to
        be evaluated at "query_points".

    Returns:
        Float[Array, "1 D"]: Point in `query_points` which maximises the utility function.
    """
    utility_function_values = utility_function(query_points)
    max_utility_function_value_idx = jnp.argmax(
        utility_function_values, axis=0, keepdims=True
    )
    best_sample_point = jnp.take_along_axis(
        query_points, max_utility_function_value_idx, axis=0
    )
    return best_sample_point


@dataclass
class AbstractSinglePointUtilityMaximizer(ABC):
    """Abstract base class for single point utility function maximizers."""

    @abstractmethod
    def maximize(
        self,
        utility_function: SinglePointUtilityFunction,
        search_space: AbstractSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        """Maximize the given utility function over the search space provided.

        Args:
            utility_function (UtilityFunction): Utility function to be
            maximized.
            search_space (AbstractSearchSpace): Search space over which to maximize
            the utility function.
            key (KeyArray): JAX PRNG key.

        Returns:
            Float[Array, "1 D"]: Point at which the utility function is maximized.
        """
        raise NotImplementedError


@dataclass
class ContinuousSinglePointUtilityMaximizer(AbstractSinglePointUtilityMaximizer):
    """The `ContinuousUtilityMaximizer` class is used to maximize utility
    functions over the continuous domain with L-BFGS-B. First we sample the utility
    function at `num_initial_samples` points from the search space, and then we run
    L-BFGS-B from the best of these initial points. We run this process `num_restarts`
    number of times, each time sampling a different random set of
    `num_initial_samples`initial points.
    """

    num_initial_samples: int
    num_restarts: int

    def __post_init__(self):
        if self.num_initial_samples < 1:
            raise ValueError(
                f"num_initial_samples must be greater than 0, got {self.num_initial_samples}."
            )
        elif self.num_restarts < 1:
            raise ValueError(
                f"num_restarts must be greater than 0, got {self.num_restarts}."
            )

    def maximize(
        self,
        utility_function: SinglePointUtilityFunction,
        search_space: ContinuousSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        max_observed_utility_function_value = None
        maximizer = None

        for _ in range(self.num_restarts):
            key, _ = jr.split(key)
            initial_sample_points = search_space.sample(
                self.num_initial_samples, key=key
            )
            best_initial_sample_point = _get_discrete_maximizer(
                initial_sample_points, utility_function
            )

            def _scalar_utility_function(x: Float[Array, "1 D"]) -> ScalarFloat:
                """
                The Jaxopt minimizer requires a function which returns a scalar. It calls the
                utility function with one point at a time, so the utility function
                returns an array of shape [1, 1], so  we index to return a scalar. Note that
                we also return the negative of the utility function - this is because
                utility functions should be *maximimized* but the Jaxopt minimizer
                minimizes functions.
                """
                return -utility_function(x)[0][0]

            lbfgsb = ScipyBoundedMinimize(
                fun=_scalar_utility_function, method="l-bfgs-b"
            )
            bounds = (search_space.lower_bounds, search_space.upper_bounds)
            optimized_point = lbfgsb.run(
                best_initial_sample_point, bounds=bounds
            ).params
            optimized_utility_function_value = _scalar_utility_function(optimized_point)
            if (max_observed_utility_function_value is None) or (
                optimized_utility_function_value > max_observed_utility_function_value
            ):
                max_observed_utility_function_value = optimized_utility_function_value
                maximizer = optimized_point
        return maximizer


AbstractUtilityMaximizer = AbstractSinglePointUtilityMaximizer
"""
Type alias for a utility maximizer. Currently we only support single point utility
functions, but in future may support batched utility functions.
"""
