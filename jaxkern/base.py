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
from typing import Callable, Dict, List, Optional, Sequence

import deprecation
import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array, Float
from jaxutils import PyTree

from .computations import AbstractKernelComputation, DenseKernelComputation


##########################################
# Abtract classes
##########################################
class AbstractKernel(PyTree):
    """
    Base kernel class"""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "AbstractKernel",
    ) -> None:
        self.compute_engine = compute_engine
        self.active_dims = active_dims
        self.stationary = stationary
        self.spectral = spectral
        self.name = name
        self.ndims = 1 if not self.active_dims else len(self.active_dims)
        compute_engine = self.compute_engine(kernel_fn=self.__call__)
        self.gram = compute_engine.gram
        self.cross_covariance = compute_engine.cross_covariance

    @abc.abstractmethod
    def __call__(
        self,
        params: Dict,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs.

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        raise NotImplementedError

    def slice_input(self, x: Float[Array, "N D"]) -> Float[Array, "N Q"]:
        """Select the relevant columns of the supplied matrix to be used within the kernel's evaluation.

        Args:
            x (Float[Array, "N D"]): The matrix or vector that is to be sliced.
        Returns:
            Float[Array, "N Q"]: A sliced form of the input matrix.
        """
        return x[..., self.active_dims]

    def __add__(self, other: "AbstractKernel") -> "AbstractKernel":
        """Add two kernels together.
        Args:
            other (AbstractKernel): The kernel to be added to the current kernel.

        Returns:
            AbstractKernel: A new kernel that is the sum of the two kernels.
        """
        return SumKernel(kernel_set=[self, other])

    def __mul__(self, other: "AbstractKernel") -> "AbstractKernel":
        """Multiply two kernels together.

        Args:
            other (AbstractKernel): The kernel to be multiplied with the current kernel.

        Returns:
            AbstractKernel: A new kernel that is the product of the two kernels.
        """
        return ProductKernel(kernel_set=[self, other])

    @property
    def ard(self):
        """Boolean property as to whether the kernel is isotropic or of
        automatic relevance determination form.

        Returns:
            bool: True if the kernel is an ARD kernel.
        """
        return True if self.ndims > 1 else False

    @abc.abstractmethod
    def init_params(self, key: KeyArray) -> Dict:
        """A template dictionary of the kernel's parameter set.

        Args:
            key (KeyArray): A PRNG key to be used for initialising
                the kernel's parameters.

        Returns:
            Dict: A dictionary of the kernel's parameters.
        """
        raise NotImplementedError

    @deprecation.deprecated(
        deprecated_in="0.0.3",
        removed_in="0.1.0",
    )
    def _initialise_params(self, key: KeyArray) -> Dict:
        """A template dictionary of the kernel's parameter set.

        Args:
            key (KeyArray): A PRNG key to be used for initialising
                the kernel's parameters.

        Returns:
            Dict: A dictionary of the kernel's parameters.
        """
        raise NotImplementedError


class CombinationKernel(AbstractKernel):
    """A base class for products or sums of kernels."""

    def __init__(
        self,
        kernel_set: List[AbstractKernel],
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "AbstractKernel",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)
        self.kernel_set = kernel_set
        name: Optional[str] = "Combination kernel"
        self.combination_fn: Optional[Callable] = None

        if not all(isinstance(k, AbstractKernel) for k in self.kernel_set):
            raise TypeError("can only combine Kernel instances")  # pragma: no cover

        self._set_kernels(self.kernel_set)

    def _set_kernels(self, kernels: Sequence[AbstractKernel]) -> None:
        """Combine multiple kernels. Based on GPFlow's Combination kernel."""
        # add kernels to a list, flattening out instances of this class therein
        kernels_list: List[AbstractKernel] = []
        for k in kernels:
            if isinstance(k, self.__class__):
                kernels_list.extend(k.kernel_set)
            else:
                kernels_list.append(k)

        self.kernel_set = kernels_list

    def init_params(self, key: KeyArray) -> Dict:
        """A template dictionary of the kernel's parameter set."""
        return [kernel.init_params(key) for kernel in self.kernel_set]

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate combination kernel on a pair of inputs.

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        return self.combination_fn(
            jnp.stack([k(p, x, y) for k, p in zip(self.kernel_set, params)])
        )


class SumKernel(CombinationKernel):
    """A kernel that is the sum of a set of kernels."""

    def __init__(
        self,
        kernel_set: List[AbstractKernel],
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Sum kernel",
    ) -> None:
        super().__init__(
            kernel_set, compute_engine, active_dims, stationary, spectral, name
        )
        self.combination_fn: Optional[Callable] = jnp.sum


class ProductKernel(CombinationKernel):
    """A kernel that is the product of a set of kernels."""

    def __init__(
        self,
        kernel_set: List[AbstractKernel],
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Product kernel",
    ) -> None:
        super().__init__(
            kernel_set, compute_engine, active_dims, stationary, spectral, name
        )
        self.combination_fn: Optional[Callable] = jnp.prod
