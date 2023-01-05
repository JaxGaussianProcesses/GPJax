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

import jaxkern
import deprecation


# These abstract types will also be removed in v0.6.0
AbstractKernel = jaxkern.kernels.AbstractKernel
AbstractKernelComputation = jaxkern.kernels.AbstractKernelComputation
CombinationKernel = jaxkern.kernels.CombinationKernel
SumKernel = jaxkern.kernels.SumKernel
ProductKernel = jaxkern.kernels.ProductKernel

# Import kernels/functions from `JaxKern`` and wrap them in a deprecation.
def deprecate(cls):
    return deprecation.deprecated(
        deprecated_in="0.5.5",
        removed_in="0.6.0",
        details="Use JaxKern for the " + cls.__name__,
    )(cls)


DiagonalKernelComputation = deprecate(jaxkern.kernels.DiagonalKernelComputation)
ConstantDiagonalKernelComputation = deprecate(
    jaxkern.kernels.ConstantDiagonalKernelComputation
)
RBF = deprecate(jaxkern.kernels.RBF)
Matern12 = deprecate(jaxkern.kernels.Matern12)
Matern32 = deprecate(jaxkern.kernels.Matern32)
Matern52 = deprecate(jaxkern.kernels.Matern52)
Linear = deprecate(jaxkern.kernels.Linear)
Periodic = deprecate(jaxkern.kernels.Periodic)
White = deprecate(jaxkern.kernels.White)
PoweredExponential = deprecate(jaxkern.kernels.PoweredExponential)
RationalQuadratic = deprecate(jaxkern.kernels.RationalQuadratic)
Polynomial = deprecate(jaxkern.kernels.Polynomial)
EigenKernelComputation = deprecate(jaxkern.kernels.EigenKernelComputation)
GraphKernel = deprecate(jaxkern.kernels.GraphKernel)
squared_distance = deprecate(jaxkern.kernels.squared_distance)
euclidean_distance = deprecate(jaxkern.kernels.euclidean_distance)
jax_gather_nd = deprecate(jaxkern.kernels.jax_gather_nd)


__all__ = [
    "AbstractKernel",
    "CombinationKernel",
    "SumKernel",
    "ProductKernel",
    "RBF",
    "Matern12",
    "Matern32",
    "Matern52",
    "Linear",
    "Periodic",
    "RationalQuadratic",
    "Polynomial",
    "White",
    "GraphKernel",
    "squared_distance",
    "euclidean_distance",
    "AbstractKernelComputation",
    "DenseKernelComputation",
    "DiagonalKernelComputation",
]
