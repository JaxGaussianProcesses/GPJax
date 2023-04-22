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

import abc
from dataclasses import dataclass
from typing import Any

from jax import vmap
from jaxtyping import Array, Float

from gpjax.linops import DenseLinearOperator, DiagonalLinearOperator, LinearOperator

Kernel = Any


@dataclass
class AbstractKernelComputation:
    """Abstract class for kernel computations."""

    kernel: Kernel

    def gram(
        self,
        x: Float[Array, "N D"],
    ) -> LinearOperator:
        """Compute Gram covariance operator of the kernel function.

        Args:
            x (Float[Array, "N N"]): The inputs to the kernel function.

        Returns
        -------
            LinearOperator: Gram covariance operator of the kernel function.
        """
        Kxx = self.cross_covariance(x, x)
        return DenseLinearOperator(Kxx)

    @abc.abstractmethod
    def cross_covariance(
        self, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        """For a given kernel, compute the NxM gram matrix on an a pair
        of input matrices with shape NxD and MxD.

        Args:
            x (Float[Array,"N D"]): The first input matrix.
            y (Float[Array,"M D"]): The second input matrix.

        Returns
        -------
            Float[Array, "N M"]: The computed cross-covariance.
        """
        raise NotImplementedError

    def diagonal(self, inputs: Float[Array, "N D"]) -> DiagonalLinearOperator:
        """For a given kernel, compute the elementwise diagonal of the
        NxN gram matrix on an input matrix of shape NxD.

        Args:
            inputs (Float[Array, "N D"]): The input matrix.

        Returns
        -------
            DiagonalLinearOperator: The computed diagonal variance entries.
        """
        return DiagonalLinearOperator(diag=vmap(lambda x: self.kernel(x, x))(inputs))
