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

from .abstractions import fit, fit_batches, fit_natgrads
from .gps import Prior, construct_posterior
from .kernels import (
    RBF,
    GraphKernel,
    Matern12,
    Matern32,
    Matern52,
    Polynomial,
    ProductKernel,
    SumKernel,
)
from .likelihoods import Bernoulli, Gaussian
from .mean_functions import Constant, Zero
from .parameters import constrain, copy_dict_structure, initialise, unconstrain
from .types import Dataset
from .variational_families import (
    CollapsedVariationalGaussian,
    ExpectationVariationalGaussian,
    NaturalVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)
from .variational_inference import CollapsedVI, StochasticVI
from . import _version

__version__ = _version.get_versions()["version"]
__license__ = "MIT"
__description__ = "Didactic Gaussian processes in JAX"
__url__ = "https://github.com/thomaspinder/GPJax"
__contributors__ = "https://github.com/thomaspinder/GPJax/graphs/contributors"


__all__ = [
    "fit",
    "fit_batches",
    "fit_natgrads",
    "Prior",
    "construct_posterior",
    "RBF",
    "GraphKernel",
    "Matern12",
    "Matern32",
    "Matern52",
    "Polynomial",
    "ProductKernel",
    "SumKernel",
    "Bernoulli",
    "Gaussian",
    "Constant",
    "Zero",
    "constrain",
    "copy_dict_structure",
    "initialise",
    "unconstrain",
    "Dataset",
    "CollapsedVariationalGaussian",
    "ExpectationVariationalGaussian",
    "NaturalVariationalGaussian",
    "VariationalGaussian",
    "WhitenedVariationalGaussian",
    "CollapsedVI",
    "StochasticVI",
]
