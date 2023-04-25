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

import jax.numpy as jnp
from jax import vmap
from gpjax.typing import Array
from jaxtyping import Float

from gpjax.linops import ConstantDiagonalLinearOperator, DiagonalLinearOperator

from .base import AbstractKernelComputation


class ConstantDiagonalKernelComputation(AbstractKernelComputation):
    def gram(self, x: Float[Array, "N D"]) -> ConstantDiagonalLinearOperator:
        """Compute Gram covariance operator of the kernel function.

        Args:
            x (Float[Array, "N N"]): The inputs to the kernel function.
        """
        value = self.kernel(x[0], x[0])

        return ConstantDiagonalLinearOperator(
            value=jnp.atleast_1d(value), size=x.shape[0]
        )

    def diagonal(self, inputs: Float[Array, "N D"]) -> DiagonalLinearOperator:
        """For a given kernel, compute the elementwise diagonal of the
        NxN gram matrix on an input matrix of shape NxD.

        Args:
            kernel (AbstractKernel): The kernel for which the variance
                vector should be computed for.
            inputs (Float[Array, "N D"]): The input matrix.

        Returns:
            LinearOperator: The computed diagonal variance entries.
        """

        diag = vmap(lambda x: self.kernel(x, x))(inputs)

        return DiagonalLinearOperator(diag=diag)

    def cross_covariance(
        self, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        """For a given kernel, compute the NxM covariance matrix on a pair of input
        matrices of shape NxD and MxD.

        Args:
            x (Float[Array,"N D"]): The input matrix.
            y (Float[Array,"M D"]): The input matrix.

        Returns:
            CovarianceOperator: The computed square Gram matrix.
        """
        # TODO: This is currently a dense implementation. We should implement a sparse LinearOperator for non-square cross-covariance matrices.
        cross_cov = vmap(lambda x: vmap(lambda y: self.kernel(x, y))(y))(x)
        return cross_cov
