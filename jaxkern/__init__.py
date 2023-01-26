# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
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

"""JaxKern."""
from .approximations import RFF
from .base import ProductKernel, SumKernel
from .computations import (
    BasisFunctionComputation,
    ConstantDiagonalKernelComputation,
    DenseKernelComputation,
    DiagonalKernelComputation,
    EigenKernelComputation,
)
from .nonstationary import Linear, Polynomial
from .stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
    Periodic,
    PoweredExponential,
    White,
)
from .non_euclidean import GraphKernel

__all__ = [
    "RBF",
    "GraphKernel",
    "Matern12",
    "Matern32",
    "Matern52",
    "Linear",
    "Polynomial",
    "ProductKernel",
    "SumKernel",
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

from . import _version

__version__ = _version.get_versions()["version"]
