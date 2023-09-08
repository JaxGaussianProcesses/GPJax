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

from beartype.typing import (
    Callable,
    Mapping,
)

from gpjax.dataset import Dataset
from gpjax.decision_making.utils import OBJECTIVE
from gpjax.gps import AbstractPosterior
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
)

SinglePointUtilityFunction = Callable[[Float[Array, "N D"]], Float[Array, "N 1"]]
"""
Type alias for utility functions which don't support batching, and instead characterise
the utility of querying a single point, rather than a batch of points. They take an array of points of shape $`[N, D]`$
and return the value of the utility function at each point in an array of shape $`[N, 1]`$.
"""


UtilityFunction = SinglePointUtilityFunction
"""
Type alias for all utility functions. Currently we only support
`SinglePointUtilityFunction`s, but in future may support batched utility functions too.
Note that `UtilityFunction`s are *maximised* in order to decide which point, or batch of points, to query next.
"""


@dataclass
class AbstractSinglePointUtilityFunctionBuilder(ABC):
    """
    Abstract class for building utility functions which don't support batches. As such,
    they characterise the utility of querying a single point next.
    """

    def check_objective_present(
        self,
        posteriors: Mapping[str, AbstractPosterior],
        datasets: Mapping[str, Dataset],
    ) -> None:
        """
        Check that the objective posterior and dataset are present in the posteriors and
        datasets.

        Args:
            posteriors (Mapping[str, AbstractPosterior]): Dictionary of posteriors to be
            used to form the utility function.
            datasets (Mapping[str, Dataset]): Dictionary of datasets which may be used
            to form the utility function.

        Raises:
            ValueError: If the objective posterior or dataset are not present in the
            posteriors or datasets.
        """
        if OBJECTIVE not in posteriors.keys():
            raise ValueError("Objective posterior not found in posteriors")
        elif OBJECTIVE not in datasets.keys():
            raise ValueError("Objective dataset not found in datasets")

    @abstractmethod
    def build_utility_function(
        self,
        posteriors: Mapping[str, AbstractPosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> SinglePointUtilityFunction:
        """
        Build a `UtilityFunction` from a set of posteriors and datasets.

        Args:
            posteriors (Mapping[str, AbstractPosterior]): Dictionary of posteriors to be
            used to form the utility function.
            datasets (Mapping[str, Dataset]): Dictionary of datasets which may be used
            to form the utility function.
            key (KeyArray): JAX PRNG key used for random number generation.

        Returns:
            SinglePointUtilityFunction: Utility function to be *maximised* in order to
            decide which point to query next.
        """
        raise NotImplementedError


AbstractUtilityFunctionBuilder = AbstractSinglePointUtilityFunctionBuilder
"""
Type alias for utility function builders. For now this only include single point utility
function builders, but in the future we may support batched utility function builders.
"""
