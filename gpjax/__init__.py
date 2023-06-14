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
from gpjax import integrators
from gpjax.base import (
    Module,
    param_field,
)
from gpjax.citation import cite
from gpjax.dataset import Dataset
from gpjax.fit import fit
from gpjax.gps import (
    Prior,
    construct_posterior,
)
from gpjax.kernels import (
    RBF,
    RFF,
    AbstractKernel,
    BasisFunctionComputation,
    ConstantDiagonalKernelComputation,
    DenseKernelComputation,
    DiagonalKernelComputation,
    EigenKernelComputation,
    GraphKernel,
    Linear,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    Polynomial,
    PoweredExponential,
    ProductKernel,
    RationalQuadratic,
    SumKernel,
    White,
)
from gpjax.likelihoods import (
    Bernoulli,
    Gaussian,
    Poisson,
)
from gpjax.mean_functions import (
    Constant,
    Zero,
)
from gpjax.objectives import (
    ELBO,
    CollapsedELBO,
    ConjugateMLL,
    LogPosteriorDensity,
    NonConjugateMLL,
)
from gpjax.variational_families import (
    CollapsedVariationalGaussian,
    ExpectationVariationalGaussian,
    NaturalVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)

__license__ = "MIT"
__description__ = "Didactic Gaussian processes in JAX"
__url__ = "https://github.com/JaxGaussianProcesses/GPJax"
__contributors__ = "https://github.com/JaxGaussianProcesses/GPJax/graphs/contributors"
__version__ = "0.0.0"

__all__ = [
    "Module",
    "param_field",
    "cite",
    "kernels",
    "fit",
    "Prior",
    "construct_posterior",
    "integrators",
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
    "Poisson",
    "Constant",
    "Zero",
    "Dataset",
    "CollapsedVariationalGaussian",
    "ExpectationVariationalGaussian",
    "NaturalVariationalGaussian",
    "VariationalGaussian",
    "WhitenedVariationalGaussian",
    "CollapsedVI",
    "StochasticVI",
    "ConjugateMLL",
    "NonConjugateMLL",
    "LogPosteriorDensity",
    "CollapsedELBO",
    "ELBO",
    "AbstractKernel",
    "Linear",
    "DenseKernelComputation",
    "DiagonalKernelComputation",
    "ConstantDiagonalKernelComputation",
    "EigenKernelComputation",
    "PoweredExponential",
    "Periodic",
    "RationalQuadratic",
    "White",
    "BasisFunctionComputation",
    "RFF",
]
