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

UtilityFunction = Callable[[Float[Array, "N D"]], Float[Array, "N 1"]]
"""
Type alias for utility functions, which take an array of points of shape $`[N, D]`$
and return the value of the utility function at each point in an array of shape $`[N, 1]`$.
"""


@dataclass
class AbstractUtilityFunctionBuilder(ABC):
    """
    Abstract class for building utility functions.
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
    ) -> UtilityFunction:
        """
        Build a `UtilityFunction` from a set of posteriors and datasets.

        Args:
            posteriors (Mapping[str, AbstractPosterior]): Dictionary of posteriors to be
            used to form the utility function.
            datasets (Mapping[str, Dataset]): Dictionary of datasets which may be used
            to form the utility function.
            key (KeyArray): JAX PRNG key used for random number generation.

        Returns:
            UtilityFunction: Utility function to be *maximised* in order to
            decide which point to query next.
        """
        raise NotImplementedError
