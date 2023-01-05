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

from jaxlinop import (
    LinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
    ConstantDiagonalLinearOperator,
)

import jax.numpy as jnp
from jax import vmap
import jax
from jaxtyping import Array, Float

from chex import PRNGKey as PRNGKeyType
from jaxutils import PyTree
import deprecation


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


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the DenseKernelComputation",
)
class DenseKernelComputation(AbstractKernelComputation):
    """Dense kernel computation class. Operations with the kernel assume
    a dense gram matrix structure.
    """

    def __init__(
        self,
        kernel_fn: Callable[
            [Dict, Float[Array, "1 D"], Float[Array, "1 D"]], Array
        ] = None,
    ) -> None:
        super().__init__(kernel_fn)

    def cross_covariance(
        self, params: Dict, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        """For a given kernel, compute the NxM covariance matrix on a pair of input
        matrices of shape NxD and MxD.

        Args:
            kernel (AbstractKernel): The kernel for which the Gram
                matrix should be computed for.
            params (Dict): The kernel's parameter set.
            x (Float[Array,"N D"]): The input matrix.
            y (Float[Array,"M D"]): The input matrix.

        Returns:
            CovarianceOperator: The computed square Gram matrix.
        """
        cross_cov = vmap(lambda x: vmap(lambda y: self.kernel_fn(params, x, y))(y))(x)
        return cross_cov


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the DiagonalKernelComputation",
)
class DiagonalKernelComputation(AbstractKernelComputation):
    def __init__(
        self,
        kernel_fn: Callable[
            [Dict, Float[Array, "1 D"], Float[Array, "1 D"]], Array
        ] = None,
    ) -> None:
        super().__init__(kernel_fn)

    def gram(
        self,
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

        diag = vmap(lambda x: self.kernel_fn(params, x, x))(inputs)

        return DiagonalLinearOperator(diag=diag)

    def cross_covariance(
        self, params: Dict, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        raise ValueError("Cross covariance not defined for diagonal kernels.")


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the ConstantDiagonalKernelComputation",
)
class ConstantDiagonalKernelComputation(AbstractKernelComputation):
    def __init__(
        self,
        kernel_fn: Callable[
            [Dict, Float[Array, "1 D"], Float[Array, "1 D"]], Array
        ] = None,
    ) -> None:
        super().__init__(kernel_fn)

    def gram(
        self,
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

        value = self.kernel_fn(params, inputs[0], inputs[0])

        return ConstantDiagonalLinearOperator(value=value, size=inputs.shape[0])

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

        diag = vmap(lambda x: self.kernel_fn(params, x, x))(inputs)

        return DiagonalLinearOperator(diag=diag)

    def cross_covariance(
        self, params: Dict, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        raise ValueError("Cross covariance not defined for constant diagonal kernels.")


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
    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """A template dictionary of the kernel's parameter set.

        Args:
            key (PRNGKeyType): A PRNG key to be used for initialising
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


##########################################
# Euclidean kernels
##########################################
@deprecation.deprecated(
    deprecated_in="0.5.5", removed_in="0.6.0", details="Use JaxKern for the RBF kernel"
)
class RBF(AbstractKernel):
    """The Radial Basis Function (RBF) kernel."""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Radial basis function kernel",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)

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
        params = {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }
        return jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x), params)


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the Matern12 kernel",
)
class Matern12(AbstractKernel):
    """The Matérn kernel with smoothness parameter fixed at 0.5."""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Matérn 1/2 kernel",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\ell` and variance :math:`\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( -\\frac{\\lvert x-y \\rvert}{2\\ell^2}  \\Bigg)

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


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the Matern32 kernel",
)
class Matern32(AbstractKernel):
    """The Matérn kernel with smoothness parameter fixed at 1.5."""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Matern 3/2",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\ell` and variance :math:`\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{3}\\lvert x-y \\rvert}{\\ell^2}  \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{3}\\lvert x-y\\rvert}{\\ell^2} \\Bigg)

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


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the Matern52 kernel",
)
class Matern52(AbstractKernel):
    """The Matérn kernel with smoothness parameter fixed at 2.5."""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Matern 5/2",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\ell` and variance :math:`\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{5}\\lvert x-y \\rvert}{\\ell^2} + \\frac{5\\lvert x - y \\rvert^2}{3\\ell^2} \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{5}\\lvert x-y\\rvert}{\\ell^2} \\Bigg)

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


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the PoweredExponential kernel",
)
class PoweredExponential(AbstractKernel):
    """The powered exponential family of kernels.

    Key reference is Diggle and Ribeiro (2007) - "Model-based Geostatistics".

    """

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Powered exponential",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)

    def __call__(self, params: dict, x: jax.Array, y: jax.Array) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\ell`, :math:`\sigma` and power :math:`\kappa`.

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( - \\Big( \\frac{\\lVert x - y \\rVert^2}{\\ell^2} \\Big)^\\kappa \\Bigg)

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.

        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * jnp.exp(-euclidean_distance(x, y) ** params["power"])
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "power": jnp.array([1.0]),
        }


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the Linear kernel",
)
class Linear(AbstractKernel):
    """The linear kernel."""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Linear",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)

    def __call__(self, params: dict, x: jax.Array, y: jax.Array) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance parameter :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 x^{T}y

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        K = params["variance"] * jnp.matmul(x.T, y)
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {"variance": jnp.array([1.0])}


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the Polynomial kernel",
)
class Polynomial(AbstractKernel):
    """The Polynomial kernel with variable degree."""

    def __init__(
        self,
        degree: int = 1,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Polynomial",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)
        self.degree = degree
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


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the White kernel",
)
class White(AbstractKernel):
    def __init__(
        self,
        compute_engine: AbstractKernelComputation = ConstantDiagonalKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "White",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)

    def __post_init__(self) -> None:
        super(White, self).__post_init__()

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


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the RationalQuadratic kernel",
)
class RationalQuadratic(AbstractKernel):
    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Rational Quadratic",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)

    def __call__(self, params: dict, x: jax.Array, y: jax.Array) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\ell` and variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( 1 + \\frac{\\lVert x - y \\rVert^2_2}{2 \\alpha \\ell^2} \\Bigg)

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * (
            1 + 0.5 * squared_distance(x, y) / params["alpha"]
        ) ** (-params["alpha"])
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "alpha": jnp.array([1.0]),
        }


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the Periodic kernel",
)
class Periodic(AbstractKernel):
    """The periodic kernel.

    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Periodic",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)

    def __call__(self, params: dict, x: jax.Array, y: jax.Array) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\ell` and variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( -0.5 \\sum_{i=1}^{d} \\Bigg)

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        sine_squared = (
            jnp.sin(jnp.pi * (x - y) / params["period"]) / params["lengthscale"]
        ) ** 2
        K = params["variance"] * jnp.exp(-0.5 * jnp.sum(sine_squared, axis=0))
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "period": jnp.array([1.0] * self.ndims),
        }


##########################################
# Graph kernels
##########################################
class EigenKernelComputation(AbstractKernelComputation):
    def __init__(
        self,
        kernel_fn: Callable[
            [Dict, Float[Array, "1 D"], Float[Array, "1 D"]], Array
        ] = None,
    ) -> None:
        super().__init__(kernel_fn)
        self._eigenvalues = None
        self._eigenvectors = None
        self._num_verticies = None

    # Define an eigenvalue setter and getter property
    @property
    def eigensystem(self) -> Float[Array, "N"]:
        return self._eigenvalues, self._eigenvectors, self._num_verticies

    @eigensystem.setter
    def eigensystem(
        self, eigenvalues: Float[Array, "N"], eigenvectors: Float[Array, "N N"]
    ) -> None:
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

    @property
    def num_vertex(self) -> int:
        return self._num_verticies

    @num_vertex.setter
    def num_vertex(self, num_vertex: int) -> None:
        self._num_verticies = num_vertex

    def _compute_S(self, params):
        evals, evecs = self.eigensystem
        S = jnp.power(
            evals
            + 2 * params["smoothness"] / params["lengthscale"] / params["lengthscale"],
            -params["smoothness"],
        )
        S = jnp.multiply(S, self.num_vertex / jnp.sum(S))
        S = jnp.multiply(S, params["variance"])
        return S

    def cross_covariance(
        self, params: Dict, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        S = self._compute_S(params=params)
        matrix = self.kernel_fn(params, x, y, S=S)
        return matrix


@deprecation.deprecated(
    deprecated_in="0.5.5", removed_in="0.6.0", details="Use JaxKern for the GraphKernel"
)
class GraphKernel(AbstractKernel):
    def __init__(
        self,
        laplacian: Float[Array, "N N"],
        compute_engine: EigenKernelComputation = EigenKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Graph kernel",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)
        self.laplacian = laplacian
        evals, self.evecs = jnp.linalg.eigh(self.laplacian)
        self.evals = evals.reshape(-1, 1)
        self.compute_engine.eigensystem = self.evals, self.evecs
        self.compute_engine.num_vertex = self.laplacian.shape[0]

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
        **kwargs,
    ) -> Float[Array, "1"]:
        """Evaluate the graph kernel on a pair of vertices :math:`v_i, v_j`.

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): Index of the ith vertex.
            y (Float[Array, "1 D"]): Index of the jth vertex.

        Returns:
            Float[Array, "1"]: The value of :math:`k(v_i, v_j)`.
        """
        S = kwargs["S"]
        Kxx = (jax_gather_nd(self.evecs, x) * S[None, :]) @ jnp.transpose(
            jax_gather_nd(self.evecs, y)
        )  # shape (n,n)
        return Kxx.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "smoothness": jnp.array([1.0]),
        }

    @property
    def num_vertex(self) -> int:
        return self.compute_engine.num_vertex


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the squared_distance function",
)
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


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the euclidean_distance function",
)
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


@deprecation.deprecated(
    deprecated_in="0.5.5",
    removed_in="0.6.0",
    details="Use JaxKern for the jax_gather_nd function",
)
def jax_gather_nd(params, indices):
    tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    return params[tuple_indices]


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
