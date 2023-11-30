# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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
from gpjax import (
    base,
    decision_making,
    gps,
    integrators,
    kernels,
    likelihoods,
    mean_functions,
    objectives,
    variational_families,
)
from gpjax.base import (
    Module,
    param_field,
)
from gpjax.citation import cite
from gpjax.dataset import Dataset
from gpjax.fit import (
    fit,
    fit_scipy,
)

__license__ = "MIT"
__description__ = "Didactic Gaussian processes in JAX"
__url__ = "https://github.com/JaxGaussianProcesses/GPJax"
__contributors__ = "https://github.com/JaxGaussianProcesses/GPJax/graphs/contributors"
__version__ = "0.8.0"

__all__ = [
    "base",
    "decision_making",
    "gps",
    "integrators",
    "kernels",
    "likelihoods",
    "mean_functions",
    "objectives",
    "variational_families",
    "Dataset",
    "cite",
    "fit",
    "Module",
    "param_field",
    "fit_scipy",
]
