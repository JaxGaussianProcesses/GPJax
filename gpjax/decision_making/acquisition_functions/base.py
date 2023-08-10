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
from typing import (
    Callable,
    Mapping,
    Optional,
)

from gpjax.dataset import Dataset
from gpjax.gps import AbstractPosterior
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
)

AcquisitionFunction = Callable[[Float[Array, "N D"]], Float[Array, "N 1"]]
"""
Type alias for acquisition functions, which take an array of points of shape $`[N, D]`$
and return the value of the acquisition function at each point in an array of shape $`[N, 1]`$.
"""


@dataclass
class AbstractAcquisitionFunctionBuilder(ABC):
    """
    Abstract class for building acquisition functions.
    """

    @abstractmethod
    def build_acquisition_function(
        self,
        posteriors: Mapping[str, AbstractPosterior],
        datasets: Optional[Mapping[str, Dataset]],
        key: KeyArray,
    ) -> AcquisitionFunction:
        """
        Build an `AcquisitionFunction` from a set of posteriors and datasets.
        """
        raise NotImplementedError
