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
from jaxopt import ScipyBoundedMinimize

from gpjax.decision_making.acquisition_functions import AcquisitionFunction
from gpjax.decision_making.search_space import (
    AbstractSearchSpace,
    ContinuousSearchSpace,
)
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
    ScalarFloat,
)


def _get_discrete_maximizer(
    query_points: Float[Array, "N D"], acquisition_function: AcquisitionFunction
) -> Float[Array, "1 D"]:
    """Get the point which maximises the acquisition function evaluated at a given set of points.

    Args:
        query_points (Float[Array, "N D"]): Set of points at which to evaluate the
        acquisition function.
        acquisition_function (AcquisitionFunction): Acquisition function
        to evaluate at `query_points`.

    Returns:
        Float[Array, "1 D"]: Point in `query_points` which maximises the acquisition
        function.
    """
    acquisition_function_values = acquisition_function(query_points)
    max_acquisition_function_value_idx = jnp.argmax(
        acquisition_function_values, axis=0, keepdims=True
    )
    best_sample_point = jnp.take_along_axis(
        query_points, max_acquisition_function_value_idx, axis=0
    )
    return best_sample_point


@dataclass
class AbstractAcquisitionMaximizer(ABC):
    """Abstract base class for acquisition function maximizers."""

    @abstractmethod
    def maximize(
        self,
        acquisition_function: AcquisitionFunction,
        search_space: AbstractSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        """Maximize the given acquisition function over the search space provided.

        Args:
            acquisition_function (AcquisitionFunction): Acquisition function to be
            maximized.
            search_space (AbstractSearchSpace): Search space over which to maximize
            the acquisition function.
            key (KeyArray): JAX PRNG key.

        Returns:
            Float[Array, "1 D"]: Point at which the acquisition function is maximized.
        """
        raise NotImplementedError


@dataclass
class ContinuousAcquisitionMaximizer(AbstractAcquisitionMaximizer):
    """The `ContinuousAcquisitionMaximizer` class is used to maximize acquisition
    functions over the continuous domain with L-BFGS-B. First we sample the acquisition
    function at `num_initial_samples` points from the search space, and then we run
    L-BFGS-B from the best of these initial points.
    """

    num_initial_samples: int

    def __post_init__(self):
        if self.num_initial_samples < 1:
            raise ValueError(
                f"num_initial_samples must be greater than 0, got {self.num_initial_samples}."
            )

    def maximize(
        self,
        acquisition_function: AcquisitionFunction,
        search_space: ContinuousSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        initial_sample_points = search_space.sample(self.num_initial_samples, key=key)
        best_initial_sample_point = _get_discrete_maximizer(
            initial_sample_points, acquisition_function
        )

        # Jaxopt minimizer requires a function which returns a scalar. It calls the
        # acquisition function with one point at a time, so the acquisition function
        # returns an array of shape [1, 1], so  we index to return a scalar. Note that
        # we also return the negative of the acquisition function - this is because
        # acquisition functions should be *maximimized* but the Jaxopt minimizer
        # minimizes functions.
        def scalar_acquisition_fn(x: Float[Array, "1 D"]) -> ScalarFloat:
            return -acquisition_function(x)[0][0]

        lbfgsb = ScipyBoundedMinimize(fun=scalar_acquisition_fn, method="l-bfgs-b")
        bounds = (search_space.lower_bounds, search_space.upper_bounds)
        optimised_point = lbfgsb.run(best_initial_sample_point, bounds=bounds).params
        return optimised_point
