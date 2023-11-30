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
import typing as tp

from cola import PSD
from cola.ops import (
    Dense,
    Diagonal,
    LinearOperator,
)
from jax import vmap
from jaxtyping import (
    Float,
    Num,
)

from gpjax.typing import Array

Kernel = tp.TypeVar("Kernel", bound="gpjax.kernels.base.AbstractKernel")  # noqa: F821


@dataclass
class AbstractKernelComputation:
    r"""Abstract class for kernel computations."""

    def gram(
        self,
        kernel: Kernel,
        x: Num[Array, "N D"],
    ) -> LinearOperator:
        r"""Compute Gram covariance operator of the kernel function.

        Args:
            kernel (AbstractKernel): the kernel function.
            x (Num[Array, "N N"]): The inputs to the kernel function.

        Returns
        -------
            LinearOperator: Gram covariance operator of the kernel function.
        """
        Kxx = self.cross_covariance(kernel, x, x)
        return PSD(Dense(Kxx))

    @abc.abstractmethod
    def cross_covariance(
        self, kernel: Kernel, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "N M"]:
        r"""For a given kernel, compute the NxM gram matrix on an a pair
        of input matrices with shape NxD and MxD.

        Args:
            kernel (AbstractKernel): the kernel function.
            x (Num[Array,"N D"]): The first input matrix.
            y (Num[Array,"M D"]): The second input matrix.

        Returns
        -------
            Float[Array, "N M"]: The computed cross-covariance.
        """
        raise NotImplementedError

    def diagonal(self, kernel: Kernel, inputs: Num[Array, "N D"]) -> Diagonal:
        r"""For a given kernel, compute the elementwise diagonal of the
        NxN gram matrix on an input matrix of shape NxD.

        Args:
            kernel (AbstractKernel): the kernel function.
            inputs (Float[Array, "N D"]): The input matrix.

        Returns
        -------
            Diagonal: The computed diagonal variance entries.
        """
        return PSD(Diagonal(diag=vmap(lambda x: kernel(x, x))(inputs)))
