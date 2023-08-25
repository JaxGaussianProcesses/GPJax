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

from beartype.typing import Mapping

from gpjax.dataset import Dataset
from gpjax.decision_making.acquisition_functions import (
    AbstractAcquisitionFunctionBuilder,
    AcquisitionFunction,
)
from gpjax.decision_making.test_functions import Quadratic
from gpjax.gps import ConjugatePosterior
from gpjax.typing import KeyArray


class QuadraticAcquisitionFunctionBuilder(AbstractAcquisitionFunctionBuilder):
    """
    Dummy acquisition function builder for testing purposes, which returns the negative
    of the value of a quadratic test function at the input points. This is because
    acquisition functions are *maximised*, and we wish to *minimise* the quadratic test
    function.
    """

    def build_acquisition_function(
        self,
        posteriors: Mapping[str, ConjugatePosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> AcquisitionFunction:
        test_function = Quadratic()
        return lambda x: -1.0 * test_function.evaluate(
            x
        )  # Acquisition functions are *maximised*
