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
from typing import Callable, Dict

from jax import vmap
from jaxlinop import (
    DenseLinearOperator,
    DiagonalLinearOperator,
    LinearOperator,
)
from jaxtyping import Array, Float
from jaxutils import PyTree


class AbstractKernelComputation(PyTree):
    """Abstract class for kernel computations."""

    def __init__(
        self,
        kernel_fn: Callable[
            [Dict, Float[Array, "1 D"], Float[Array, "1 D"]], Array
        ] = None,
    ) -> None:
        self._kernel_fn = kernel_fn

    @property
    def kernel_fn(
        self,
    ) -> Callable[[Dict, Float[Array, "1 D"], Float[Array, "1 D"]], Array]:
        return self._kernel_fn

    @kernel_fn.setter
    def kernel_fn(
        self,
        kernel_fn: Callable[[Dict, Float[Array, "1 D"], Float[Array, "1 D"]], Array],
    ) -> None:
        self._kernel_fn = kernel_fn

    def gram(
        self,
        params: Dict,
        inputs: Float[Array, "N D"],
    ) -> LinearOperator:

        """Compute Gram covariance operator of the kernel function.

        Args:
            kernel (AbstractKernel): The kernel function to be evaluated.
            params (Dict): The parameters of the kernel function.
            inputs (Float[Array, "N N"]): The inputs to the kernel function.

        Returns:
            LinearOperator: Gram covariance operator of the kernel function.
        """

        matrix = self.cross_covariance(params, inputs, inputs)

        return DenseLinearOperator(matrix=matrix)

    @abc.abstractmethod
    def cross_covariance(
        self,
        params: Dict,
        x: Float[Array, "N D"],
        y: Float[Array, "M D"],
    ) -> Float[Array, "N M"]:
        """For a given kernel, compute the NxM gram matrix on an a pair
        of input matrices with shape NxD and MxD.

        Args:
            kernel (AbstractKernel): The kernel for which the cross-covariance
                matrix should be computed for.
            params (Dict): The kernel's parameter set.
            x (Float[Array,"N D"]): The first input matrix.
            y (Float[Array,"M D"]): The second input matrix.

        Returns:
            Float[Array, "N M"]: The computed square Gram matrix.
        """
        raise NotImplementedError

    def diagonal(
        self,
        params: Dict,
        inputs: Float[Array, "N D"],
    ) -> DiagonalLinearOperator:
        """For a given kernel, compute the elementwise diagonal of the
        NxN gram matrix on an input matrix of shape NxD.

        Args:
            kernel (AbstractKernel): The kernel for which the variance
                vector should be computed for.
            params (Dict): The kernel's parameter set.
            inputs (Float[Array, "N D"]): The input matrix.

        Returns:
            LinearOperator: The computed diagonal variance entries.
        """
        diag = vmap(lambda x: self._kernel_fn(params, x, x))(inputs)

        return DiagonalLinearOperator(diag=diag)
