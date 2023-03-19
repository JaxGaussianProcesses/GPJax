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

import jaxkern as kernels
import jaxutils as utils

from . import _version
from .gps import Prior, construct_posterior
from .likelihoods import Bernoulli, Gaussian
from .mean_functions import Constant, Zero
from .objectives import ConjugateMLL, NonConjugateMLL, ELBO, CollapsedELBO
from .variational_families import (
    CollapsedVariationalGaussian,
    ExpectationVariationalGaussian,
    NaturalVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)
from .variational_inference import CollapsedVI, StochasticVI

__version__ = _version.get_versions()["version"]
__license__ = "MIT"
__description__ = "Didactic Gaussian processes in JAX"
__url__ = "https://github.com/JAXGaussianProcesses/GPJax"
__contributors__ = "https://github.com/JAXGaussianProcesses/GPJax/graphs/contributors"


__all__ = [
    "Prior",
    "construct_posterior",
    "ConjugateMLL",
    "NonConjugateMLL",
    "ELBO",
    "CollapsedELBO",
    "Bernoulli",
    "Gaussian",
    "Constant",
    "Zero",
    "CollapsedVariationalGaussian",
    "ExpectationVariationalGaussian",
    "NaturalVariationalGaussian",
    "VariationalGaussian",
    "WhitenedVariationalGaussian",
    "CollapsedVI",
    "StochasticVI",
    "kernels",
    "utils",
]
