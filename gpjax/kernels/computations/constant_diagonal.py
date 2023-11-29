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

import typing as tp

from cola import PSD
from cola.ops import (
    Diagonal,
    Identity,
    LinearOperator,
)
from jax import vmap
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.typing import Array

Kernel = tp.TypeVar("Kernel", bound="gpjax.kernels.base.AbstractKernel")  # noqa: F821


class ConstantDiagonalKernelComputation(AbstractKernelComputation):
    def gram(self, kernel: Kernel, x: Float[Array, "N D"]) -> LinearOperator:
        r"""Compute the Gram matrix.

        Compute Gram covariance operator of the kernel function.

        Args:
            kernel (Kernel): the kernel function.
            x (Float[Array, "N D"]): The inputs to the kernel function.

        Returns
        -------
            LinearOperator: Gram covariance operator of the kernel function.
        """
        value = kernel(x[0], x[0])
        dtype = value.dtype
        shape = (x.shape[0], x.shape[0])

        return PSD(jnp.atleast_1d(value) * Identity(shape=shape, dtype=dtype))

    def diagonal(self, kernel: Kernel, inputs: Float[Array, "N D"]) -> Diagonal:
        r"""Compute the diagonal Gram matrix's entries.

        For a given kernel, compute the elementwise diagonal of the
        NxN gram matrix on an input matrix of shape $`N\times D`$.

        Args:
            kernel (Kernel): the kernel function.
            inputs (Float[Array, "N D"]): The input matrix.

        Returns
        -------
            Diagonal: The computed diagonal variance entries.
        """
        diag = vmap(lambda x: kernel(x, x))(inputs)

        return PSD(Diagonal(diag=diag))

    def cross_covariance(
        self, kernel: Kernel, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        r"""Compute the cross-covariance matrix.

        For a given kernel, compute the NxM covariance matrix on a pair of input
        matrices of shape NxD and MxD.

        Args:
            kernel (Kernel): the kernel function.
            x (Float[Array,"N D"]): The input matrix.
            y (Float[Array,"M D"]): The input matrix.

        Returns
        -------
            Float[Array, "N M"]: The computed square Gram matrix.
        """
        # TODO: This is currently a dense implementation. We should implement
        # a sparse LinearOperator for non-square cross-covariance matrices.
        cross_cov = vmap(lambda x: vmap(lambda y: kernel(x, y))(y))(x)
        return cross_cov
