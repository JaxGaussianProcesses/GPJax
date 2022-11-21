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

import abc
from typing import Callable, Dict, List, Optional, Sequence

from jaxlinop import(
    LinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
    ConstantDiagonalLinearOperator,
)

import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jaxtyping import Array, Float

from .config import get_defaults
from .types import PRNGKeyType

JITTER = get_defaults()["jitter"]

##########################################
# Abtract classes
##########################################
@dataclass(repr=False)
class AbstractKernel:
    """
    Base kernel class"""

    active_dims: Optional[List[int]] = None
    stationary: Optional[bool] = False
    spectral: Optional[bool] = False
    name: Optional[str] = "AbstractKernel"

    def __post_init__(self) -> None:
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

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
    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """A template dictionary of the kernel's parameter set.

        Args:
            key (PRNGKeyType): A PRNG key to be used for initialising
                the kernel's parameters.

        Returns:
            Dict: A dictionary of the kernel's parameters.
        """
        raise NotImplementedError


@dataclass
class AbstractKernelComputation:
    """Abstract class for kernel computations."""

    @staticmethod
    @abc.abstractmethod
    def gram(
        kernel: AbstractKernel,
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

        raise NotImplementedError

    @staticmethod
    def cross_covariance(
        kernel: AbstractKernel,
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

        cross_cov = vmap(lambda x: vmap(lambda y: kernel(params, x, y))(y))(x)

        return cross_cov

    @staticmethod
    def diagonal(
        kernel: AbstractKernel,
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

        diag = vmap(lambda x: kernel(params, x, x))(inputs)

        return DiagonalLinearOperator(diag=diag)


class DenseKernelComputation(AbstractKernelComputation):
    """Dense kernel computation class. Operations with the kernel assume
    a dense gram matrix structure.
    """

    @staticmethod
    def gram(
        kernel: AbstractKernel,
        params: Dict,
        inputs: Float[Array, "N D"],
    ) -> DenseLinearOperator:
        """For a given kernel, compute the NxN gram matrix on an input
        matrix of shape NxD.

        Args:
            kernel (AbstractKernel): The kernel for which the Gram
                matrix should be computed for.
            params (Dict): The kernel's parameter set.
            inputs (Float[Array,"N D"]): The input matrix.

        Returns:
            CovarianceOperator: The computed square Gram matrix.
        """

        matrix = vmap(lambda x: vmap(lambda y: kernel(params, x, y))(inputs))(inputs)

        return DenseLinearOperator(matrix=matrix)


class DiagonalKernelComputation(AbstractKernelComputation):
    @staticmethod
    def gram(
        kernel: AbstractKernel,
        params: Dict,
        inputs: Float[Array, "N D"],
    ) -> DiagonalLinearOperator:
        """For a kernel with diagonal structure, compute the NxN gram matrix on
        an input matrix of shape NxD.

        Args:
            kernel (AbstractKernel): The kernel for which the Gram matrix
                should be computed for.
            params (Dict): The kernel's parameter set.
            inputs (Float[Array, "N D"]): The input matrix.

        Returns:
            CovarianceOperator: The computed square Gram matrix.
        """

        diag = vmap(lambda x: kernel(params, x, x))(inputs)

        return DiagonalLinearOperator(diag=diag)


class ConstantDiagonalKernelComputation(AbstractKernelComputation):
    @staticmethod
    def gram(
        kernel: AbstractKernel,
        params: Dict,
        inputs: Float[Array, "N D"],
    ) -> ConstantDiagonalLinearOperator:
        """For a kernel with diagonal structure, compute the NxN gram matrix on
        an input matrix of shape NxD.

        Args:
            kernel (AbstractKernel): The kernel for which the Gram matrix
                should be computed for.
            params (Dict): The kernel's parameter set.
            inputs (Float[Array, "N D"]): The input matrix.

        Returns:
            CovarianceOperator: The computed square Gram matrix.
        """

        value = kernel(params, inputs[0], inputs[0])

        return ConstantDiagonalLinearOperator(value=value, size=inputs.shape[0])


    @staticmethod
    def diagonal(
        kernel: AbstractKernel,
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

        diag = vmap(lambda x: kernel(params, x, x))(inputs)

        return DiagonalLinearOperator(diag=diag)


@dataclass
class _KernelSet:
    """A mixin class for storing a list of kernels. Useful for combination kernels."""

    kernel_set: List[AbstractKernel]


@dataclass
class CombinationKernel(AbstractKernel, _KernelSet, DenseKernelComputation):
    """A base class for products or sums of kernels."""

    name: Optional[str] = "Combination kernel"
    combination_fn: Optional[Callable] = None

    def __post_init__(self) -> None:
        """Set the kernel set to the list of kernels passed to the constructor."""
        kernels = self.kernel_set

        if not all(isinstance(k, AbstractKernel) for k in kernels):
            raise TypeError("can only combine Kernel instances")  # pragma: no cover

        self.kernel_set: List[AbstractKernel] = []
        self._set_kernels(kernels)

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

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """A template dictionary of the kernel's parameter set."""
        return [kernel._initialise_params(key) for kernel in self.kernel_set]

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


@dataclass
class SumKernel(CombinationKernel):
    """A kernel that is the sum of a set of kernels."""

    name: Optional[str] = "Sum kernel"
    combination_fn: Optional[Callable] = jnp.sum


@dataclass
class ProductKernel(CombinationKernel):
    """A kernel that is the product of a set of kernels."""

    name: Optional[str] = "Product kernel"
    combination_fn: Optional[Callable] = jnp.prod


##########################################
# Euclidean kernels
##########################################
@dataclass(repr=False)
class RBF(AbstractKernel, DenseKernelComputation):
    """The Radial Basis Function (RBF) kernel."""

    name: Optional[str] = "Radial basis function kernel"

    def __post_init__(self) -> None:
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\ell` and variance :math:`\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( \\frac{\\lVert x - y \\rVert^2_2}{2 \\ell^2} \\Bigg)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


@dataclass(repr=False)
class Matern12(AbstractKernel, DenseKernelComputation):
    """The Matérn kernel with smoothness parameter fixed at 0.5."""

    name: Optional[str] = "Matern 1/2"

    def __post_init__(self) -> None:
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\ell` and variance :math:`\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( -\\frac{\\lvert x-y \\rvert}{\\ell}  \\Bigg)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call
        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * jnp.exp(-euclidean_distance(x, y))
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


@dataclass(repr=False)
class Matern32(AbstractKernel, DenseKernelComputation):
    """The Matérn kernel with smoothness parameter fixed at 1.5."""

    name: Optional[str] = "Matern 3/2"

    def __post_init__(self) -> None:
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\ell` and variance :math:`\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{3}\\lvert x-y \\rvert}{\\ell}  \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{3}\\lvert x-y\\rvert}{\\ell} \\Bigg)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        tau = euclidean_distance(x, y)
        K = (
            params["variance"]
            * (1.0 + jnp.sqrt(3.0) * tau)
            * jnp.exp(-jnp.sqrt(3.0) * tau)
        )
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


@dataclass(repr=False)
class Matern52(AbstractKernel, DenseKernelComputation):
    """The Matérn kernel with smoothness parameter fixed at 2.5."""

    name: Optional[str] = "Matern 5/2"

    def __post_init__(self) -> None:
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\ell` and variance :math:`\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{5}\\lvert x-y \\rvert}{\\ell} + \\frac{5\\lvert x - y \\rvert^2}{3\\ell^2} \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{5}\\lvert x-y\\rvert}{\\ell} \\Bigg)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        tau = euclidean_distance(x, y)
        K = (
            params["variance"]
            * (1.0 + jnp.sqrt(5.0) * tau + 5.0 / 3.0 * jnp.square(tau))
            * jnp.exp(-jnp.sqrt(5.0) * tau)
        )
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


@dataclass(repr=False)
class Polynomial(AbstractKernel, DenseKernelComputation):
    """The Polynomial kernel with variable degree."""

    name: Optional[str] = "Polynomial"
    degree: int = 1

    def __post_init__(self) -> None:
        self.ndims = 1 if not self.active_dims else len(self.active_dims)
        self.name = f"Polynomial Degree: {self.degree}"

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with shift parameter :math:`\\alpha` and variance :math:`\sigma^2` through

        .. math::
            k(x, y) = \\Big( \\alpha + \\sigma^2 xy \\Big)^{d}

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x).squeeze()
        y = self.slice_input(y).squeeze()
        K = jnp.power(params["shift"] + jnp.dot(x * params["variance"], y), self.degree)
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {
            "shift": jnp.array([1.0]),
            "variance": jnp.array([1.0] * self.ndims),
        }


@dataclass(repr=False)
class White(AbstractKernel, ConstantDiagonalKernelComputation):
    def __post_init__(self) -> None:
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \delta(x-y)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        K = jnp.all(jnp.equal(x, y)) * params["variance"]
        return K.squeeze()

    def _initialise_params(self, key: Float[Array, "1 D"]) -> Dict:
        """Initialise the kernel parameters.

        Args:
            key (Float[Array, "1 D"]): The key to initialise the parameters with.

        Returns:
            Dict: The initialised parameters.
        """
        return {"variance": jnp.array([1.0])}


##########################################
# Graph kernels
##########################################
@dataclass
class _EigenKernel:
    laplacian: Float[Array, "N N"]


@dataclass
class GraphKernel(AbstractKernel, _EigenKernel, DenseKernelComputation):
    name: Optional[str] = "Graph kernel"

    def __post_init__(self) -> None:
        self.ndims = 1
        evals, self.evecs = jnp.linalg.eigh(self.laplacian)
        self.evals = evals.reshape(-1, 1)
        self.num_vertex = self.laplacian.shape[0]

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the graph kernel on a pair of vertices :math:`v_i, v_j`.

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): Index of the ith vertex.
            y (Float[Array, "1 D"]): Index of the jth vertex.

        Returns:
            Float[Array, "1"]: The value of :math:`k(v_i, v_j)`.
        """
        psi = jnp.power(
            2 * params["smoothness"] / params["lengthscale"] ** 2 + self.evals,
            -params["smoothness"],
        )
        psi *= self.num_vertex / jnp.sum(psi)
        x_evec = self.evecs[:, x]
        y_evec = self.evecs[:, y]
        kxy = params["variance"] * jnp.sum(
            jnp.prod(jnp.stack([psi, x_evec, y_evec]).squeeze(), axis=0)
        )
        return kxy.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "smoothness": jnp.array([1.0]),
        }


def squared_distance(
    x: Float[Array, "1 D"], y: Float[Array, "1 D"]
) -> Float[Array, "1"]:
    """Compute the squared distance between a pair of inputs.

    Args:
        x (Float[Array, "1 D"]): First input.
        y (Float[Array, "1 D"]): Second input.

    Returns:
        Float[Array, "1"]: The squared distance between the inputs.
    """

    return jnp.sum((x - y) ** 2)


def euclidean_distance(
    x: Float[Array, "1 D"], y: Float[Array, "1 D"]
) -> Float[Array, "1"]:
    """Compute the euclidean distance between a pair of inputs.

    Args:
        x (Float[Array, "1 D"]): First input.
        y (Float[Array, "1 D"]): Second input.

    Returns:
        Float[Array, "1"]: The euclidean distance between the inputs.
    """

    return jnp.sqrt(jnp.maximum(squared_distance(x, y), 1e-36))


__all__ = [
    "AbstractKernel",
    "CombinationKernel",
    "SumKernel",
    "ProductKernel",
    "RBF",
    "Matern12",
    "Matern32",
    "Matern52",
    "Polynomial",
    "White",
    "GraphKernel",
    "squared_distance",
    "euclidean_distance",
    "AbstractKernelComputation",
    "DenseKernelComputation",
    "DiagonalKernelComputation",
]
