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


import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)

import gpjax  # noqa: F401
from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.typing import Array

Kernel = tp.TypeVar(
    "Kernel",
    bound="gpjax.kernels.non_euclidean.graph.GraphKernel",  # noqa: F821
)


class EigenKernelComputation(AbstractKernelComputation):
    r"""Eigen kernel computation class. Kernels who operate on an
    eigen-decomposed structure should use this computation object.
    """

    def _cross_covariance(
        self, kernel: Kernel, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "N M"]:
        # Transform the eigenvalues of the graph Laplacian according to the
        # RBF kernel's SPDE form.
        S = jnp.power(
            kernel.eigenvalues.value
            + 2
            * kernel.smoothness.value
            / kernel.lengthscale.value
            / kernel.lengthscale.value,
            -kernel.smoothness.value,
        )
        S = jnp.multiply(S, kernel.num_vertex / jnp.sum(S))
        # Scale the transform eigenvalues by the kernel variance
        S = jnp.multiply(S, kernel.variance.value)
        return kernel(x, y, S=S)
