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
from cola import PSD
from cola.ops import (
    Diagonal,
    LinearOperator,
)
from jax import vmap
from jaxtyping import Float

from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.typing import Array

Kernel = tp.TypeVar("Kernel", bound="gpjax.kernels.base.AbstractKernel")  # noqa: F821


class DiagonalKernelComputation(AbstractKernelComputation):
    r"""Diagonal kernel computation class. Operations with the kernel assume
    a diagonal Gram matrix.
    """

    def gram(self, kernel: Kernel, x: Float[Array, "N D"]) -> LinearOperator:
        r"""Compute the Gram matrix.

        For a kernel with diagonal structure, compute the $`N\times N`$ Gram matrix on
        an input matrix of shape $`N\times D`$.

        Args:
            kernel (Kernel): the kernel function.
            x (Float[Array, "N D"]): The input matrix.

        Returns
        -------
            LinearOperator: The computed square Gram matrix.
        """
        return PSD(Diagonal(diag=vmap(lambda x: kernel(x, x))(x)))

    def cross_covariance(
        self, kernel: Kernel, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        r"""Compute the cross-covariance matrix.

        For a given kernel, compute the $`N\times M`$ covariance matrix on a pair of
        input matrices of shape $`N\times D`$ and $`M\times D`$.

        Args:
            kernel (Kernel): the kernel function.
            x (Float[Array,"N D"]): The input matrix.
            y (Float[Array,"M D"]): The input matrix.

        Returns
        -------
            Float[Array, "N M"]: The computed cross-covariance.
        """
        # TODO: This is currently a dense implementation. We should implement a sparse LinearOperator for non-square cross-covariance matrices.
        cross_cov = vmap(lambda x: vmap(lambda y: kernel(x, y))(y))(x)
        return cross_cov
