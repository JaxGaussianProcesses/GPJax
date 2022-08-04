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
from functools import reduce
from typing import Callable, Dict, List, Optional, Sequence

import jax.numpy as jnp
import jax.scipy as jsp
from chex import dataclass
from jax import vmap
from jaxtyping import Array, Float

from .types import PRNGKeyType

from .config import get_defaults
from .utils import I

JITTER = get_defaults()["jitter"]

##########################################
# Abtract classes
##########################################
@dataclass(repr=False)
class Kernel:
    """Base kernel class"""

    active_dims: Optional[List[int]] = None
    stationary: Optional[bool] = False
    spectral: Optional[bool] = False
    name: Optional[str] = "Kernel"

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    @abc.abstractmethod
    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"], params: Dict
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs.
        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (Dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`.
        """
        raise NotImplementedError

    def slice_input(self, x: Float[Array, "N D"]) -> Float[Array, "N Q"]:
        """Select the relevant columns of the supplied matrix to be used within the kernel's evaluation.
        Args:
            x (Array): The matrix or vector that is to be sliced.
        Returns:
            Array: A sliced form of the input matrix.
        """
        return x[..., self.active_dims]

    def __add__(self, other: "Kernel") -> "Kernel":
        return SumKernel(kernel_set=[self, other])

    def __mul__(self, other: "Kernel") -> "Kernel":
        return ProductKernel(kernel_set=[self, other])

    def __matmul__(self, other: "Kernel") -> "Kernel":
        return KroneckerKernel(kernel_set=[self, other])

    @property
    def ard(self):
        """Boolean property as to whether the kernel is isotropic or of automatic relevance determination form.

        Returns:
            bool: True if the kernel is an ARD kernel.
        """
        return True if self.ndims > 1 else False

    @abc.abstractmethod
    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """A template dictionary of the kernel's parameter set."""
        raise NotImplementedError


@dataclass
class AbstractKernelComputation:
    @abc.abstractmethod
    def gram():
        raise NotImplementedError

    @abc.abstractmethod
    def diagonal():
        raise NotImplementedError

    @abc.abstractmethod
    def cross_covariance():
        raise NotImplementedError

    def solve():
        raise NotImplementedError


class DenseKernelComputation(AbstractKernelComputation):
    @staticmethod
    def gram(kernel: Kernel, inputs: f64["N D"], params: dict) -> f64["N N"]:
        """For a given kernel, compute the :math:`n \times n` gram matrix on an input matrix of shape :math:`n \times d` for :math:`d\geq 1`.

        Args:
            kernel (Kernel): The kernel for which the Gram matrix should be computed for.
            inputs (Array): The input matrix.
            params (dict): The kernel's parameter set.

        Returns:
            Array: The computed square Gram matrix.
        """
        return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(inputs))(inputs)

    @staticmethod
    def cross_covariance(
        kernel: Kernel, x: f64["N D"], y: f64["M D"], params: dict
    ) -> f64["N M"]:
        """For a given kernel, compute the :math:`m \times n` gram matrix on an a pair of input matrices with shape :math:`m \times d`  and :math:`n \times d` for :math:`d\geq 1`.

        Args:
            kernel (Kernel): The kernel for which the cross-covariance matrix should be computed for.
            x (Array): The first input matrix.
            y (Array): The second input matrix.
            params (dict): The kernel's parameter set.

        Returns:
            Array: The computed square Gram matrix.
        """
        return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(y))(x)

    @staticmethod
    def diagonal(kernel: Kernel, inputs: f64["N D"], params: dict) -> f64["N N"]:
        """For a given kernel, compute the elementwise diagonal of the :math:`n \times n` gram matrix on an input matrix of shape :math:`n \times d` for :math:`d\geq 1`.
        Args:
            kernel (Kernel): The kernel for which the variance vector should be computed for.
            inputs (Array): The input matrix.
            params (dict): The kernel's parameter set.
        Returns:
            Array: The computed diagonal variance matrix.
        """
        return jnp.diag(vmap(lambda x: kernel(x, x, params))(inputs))


@dataclass
class AbstractKernelComputation:
    @abc.abstractmethod
    def gram():
        raise NotImplementedError

    @abc.abstractmethod
    def diagonal():
        raise NotImplementedError

    @abc.abstractmethod
    def cross_covariance():
        raise NotImplementedError

    @abc.abstractmethod
    def gram_solve():
        raise NotImplementedError


class DenseKernelComputation(AbstractKernelComputation):
    @staticmethod
    def gram(kernel: Kernel, inputs: f64["N D"], params: dict) -> f64["N N"]:
        """For a given kernel, compute the :math:`n \times n` gram matrix on an input matrix of shape :math:`n \times d` for :math:`d\geq 1`.

        Args:
            kernel (Kernel): The kernel for which the Gram matrix should be computed for.
            inputs (Array): The input matrix.
            params (dict): The kernel's parameter set.

        Returns:
            Array: The computed square Gram matrix.
        """
        return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(inputs))(inputs)

    @staticmethod
    def gram_multiply(kernel, inputs, params: dict, array: f64["N"]):
        """
        Compute the matrix multiplication of the Gram matrix with an array.
        """
        pass

    @staticmethod
    def cross_covariance(
        kernel: Kernel, x: f64["N D"], y: f64["M D"], params: dict
    ) -> f64["N M"]:
        """For a given kernel, compute the :math:`m \times n` gram matrix on an a pair of input matrices with shape :math:`m \times d`  and :math:`n \times d` for :math:`d\geq 1`.

        Args:
            kernel (Kernel): The kernel for which the cross-covariance matrix should be computed for.
            x (Array): The first input matrix.
            y (Array): The second input matrix.
            params (dict): The kernel's parameter set.

        Returns:
            Array: The computed square Gram matrix.
        """
        return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(y))(x)

    @staticmethod
    def diagonal(kernel: Kernel, inputs: f64["N D"], params: dict) -> f64["N"]:
        """For a given kernel, compute the elementwise diagonal of the :math:`n \times n` gram matrix on an input matrix of shape :math:`n \times d` for :math:`d\geq 1`.
        Args:
            kernel (Kernel): The kernel for which the variance vector should be computed for.
            inputs (Array): The input matrix.
            params (dict): The kernel's parameter set.
        Returns:
            Array: The computed diagonal variance entries.
        """
        return vmap(lambda x: kernel(x, x, params))(inputs)

    @staticmethod
    def gram_solve(
        kernel: Kernel, inputs: f64["N D"], params: dict, array: f64["N"]
    ) -> f64["N N"]:
        """For a given kernel, compute the inverse :math:`m \times n` gram matrix on an a pair of input matrices with shape :math:`m \times d`  and :math:`n \times d` for :math:`d\geq 1`.

        Args:
            kernel (Kernel): The kernel for which the cross-covariance matrix should be computed for.
            x (Array): The first input matrix.
            y (Array): The second input matrix.
            params (dict): The kernel's parameter set.

        Returns:
            Array: The computed inverse square Gram matrix.
        """
        L = kernel.gram_cholesky(kernel, inputs, params)

        return jsp.linalg.solve_triangular(
            L.T, jsp.linalg.solve_triangular(L, array, lower=True), lower=True
        )

    @staticmethod
    def gram_cholesky(kernel: Kernel, inputs: f64["N D"], params: dict) -> f64["N N"]:
        """Compute the Cholesky decomposition of the :math:`n \times n` gram matrix on an input matrix of shape :math:`n \times d` for :math:`d\geq 1`.
        Args:
            kernel (Kernel): The kernel for which the Cholesky decomposition should be computed for.
            inputs (Array): The input matrix.
            params (dict): The kernel's parameter set.
        Returns:
            Array: The computed Cholesky decomposition.
        """
        n = inputs.shape[0]
        Kxx = vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(inputs))(inputs)
        return jnp.linalg.cholesky(Kxx + I(n) * JITTER)


class DiagonalKernelComputation(AbstractKernelComputation):
    @staticmethod
    def gram(kernel: Kernel, inputs: f64["N D"], params: dict) -> f64["N N"]:
        """For a given kernel, compute the :math:`n \times n` gram matrix on an input matrix of shape :math:`n \times d` for :math:`d\geq 1`.

        Args:
            kernel (Kernel): The kernel for which the Gram matrix should be computed for.
            inputs (Array): The input matrix.
            params (dict): The kernel's parameter set.

        Returns:
            Array: The computed square Gram matrix.
        """
        return jnp.diag(vmap(lambda x: kernel(x, x, params))(inputs))

    @staticmethod
    def gram_multiply(kernel, inputs, params: dict, array: f64["N"]):
        """
        Compute the matrix multiplication of the Gram matrix with an array.
        """
        pass

    @staticmethod
    def cross_covariance(
        kernel: Kernel, x: f64["N D"], y: f64["M D"], params: dict
    ) -> f64["N M"]:
        """For a given kernel, compute the :math:`m \times n` gram matrix on an a pair of input matrices with shape :math:`m \times d`  and :math:`n \times d` for :math:`d\geq 1`.

        Args:
            kernel (Kernel): The kernel for which the cross-covariance matrix should be computed for.
            x (Array): The first input matrix.
            y (Array): The second input matrix.
            params (dict): The kernel's parameter set.

        Returns:
            Array: The computed square Gram matrix.
        """
        return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(y))(x)

    @staticmethod
    def diagonal(kernel: Kernel, inputs: f64["N D"], params: dict) -> f64["N N"]:
        """For a given kernel, compute the elementwise diagonal of the :math:`n \times n` gram matrix on an input matrix of shape :math:`n \times d` for :math:`d\geq 1`.
        Args:
            kernel (Kernel): The kernel for which the variance vector should be computed for.
            inputs (Array): The input matrix.
            params (dict): The kernel's parameter set.
        Returns:
            Array: The computed diagonal variance matrix.
        """
        return vmap(lambda x: kernel(x, x, params))(inputs)

    @staticmethod
    def gram_solve(
        kernel: Kernel, inputs: f64["N D"], params: dict, array: f64["N"]
    ) -> f64["N"]:
        """For a given kernel, compute the inverse :math:`m \times n` gram matrix on an a pair of input matrices with shape :math:`m \times d`  and :math:`n \times d` for :math:`d\geq 1`.

        Args:
            kernel (Kernel): The kernel for which the cross-covariance matrix should be computed for.
            x (Array): The first input matrix.
            y (Array): The second input matrix.
            params (dict): The kernel's parameter set.

        Returns:
            Array: The computed inverse square Gram matrix.
        """

        return array / vmap(lambda x: kernel(x, x, params))(inputs)

    @staticmethod
    def gram_cholesky(kernel: Kernel, inputs: f64["N D"], params: dict) -> f64["N N"]:
        """Compute the Cholesky decomposition of the :math:`n \times n` gram matrix on an input matrix of shape :math:`n \times d` for :math:`d\geq 1`.
        Args:
            kernel (Kernel): The kernel for which the Cholesky decomposition should be computed for.
            inputs (Array): The input matrix.
            params (dict): The kernel's parameter set.
        Returns:
            Array: The computed Cholesky decomposition.
        """
        return jnp.diag(jnp.sqrt(kernel.diagonal(kernel, inputs, params)))


@dataclass
class _KernelSet:
    kernel_set: List[Kernel]


@dataclass
class CombinationKernel(Kernel, _KernelSet):
    """A base class for products or sums of kernels."""

    name: Optional[str] = "Combination kernel"
    combination_fn: Optional[Callable] = None

    def __post_init__(self):
        """Set the kernel set to the list of kernels passed to the constructor."""
        kernels = self.kernel_set

        if not all(isinstance(k, Kernel) for k in kernels):
            raise TypeError("can only combine Kernel instances")  # pragma: no cover

        self.kernel_set: List[Kernel] = []
        self._set_kernels(kernels)

        # Temporary solution for promoting computation types in combination kernels (does not work in all cases!).
        if all(list(map(lambda k: isinstance(k, DiagonalKernelComputation), kernels))):
            computation = DiagonalKernelComputation()

        else:
            computation = DenseKernelComputation()

        self._gram = computation.gram
        self._cross_covariance = computation.cross_covariance
        self._diagonal = computation.diagonal
        self._gram_solve = computation.gram_solve
        self._gram_cholesky = computation.gram_cholesky

    @property
    def gram(self) -> Callable:
        return self._gram

    @property
    def cross_covariance(self) -> Callable:
        return self._cross_covariance

    @property
    def diagonal(self) -> Callable:
        return self._diagonal

    @property
    def gram_solve(self) -> Callable:
        return self._gram_solve

    @property
    def gram_cholesky(self) -> Callable:
        return self._gram_cholesky

    def _set_kernels(self, kernels: Sequence[Kernel]) -> None:
        """Combine multiple kernels. Based on GPFlow's Combination kernel."""
        # add kernels to a list, flattening out instances of this class therein
        kernels_list: List[Kernel] = []
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
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"], params: Dict
    ) -> Float[Array, "1"]:
        return self.combination_fn(
            jnp.stack([k(x, y, p) for k, p in zip(self.kernel_set, params)])
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

    # TODO: Check active dims of the kernels - i.e., are they separable?
    #       If so then we can use the cheaper Krocker computation backend.
    # Obviously, this should be done on the main 'Kernel' class.
    # On second thought, this would not be (currently) possible - since we need a grid design space.
    # Could this perhaps be resolved via evaluating the kernel on a fake grid, introducing a matrix W,
    # comprising a subsample of the indentity matrix and zeros at missing points -> compute W @ K @ W^{T} ?


class KroneckerKernelComputation(AbstractKernelComputation):
    @staticmethod
    def gram(kernel, inputs, params: dict):
        gram_matrices = []

        # --- TODO: rewrite (--- to ---) in a more JAX-friendly way!
        for k, params in zip(kernel.kernel_set, params):

            # Filter grid based on active dims.
            _, indices = jnp.unique(
                inputs[..., k.active_dims], return_index=True, axis=0
            )

            gram_matrices.append(k.gram(k, inputs[indices], params))
        # ---

        return reduce(jnp.kron, gram_matrices)

    @staticmethod
    def gram_multiply(kernel, inputs, params: dict, array: f64["N"]):
        """
        Compute the matrix multiplication of the Gram matrix with an array.
        """
        pass

    # TODO: Do we save anything for commputing the cross covariance with Kronecker kernels?
    # @staticmethod
    # def cross_covariance(kernel, inputs, params: dict):
    #     pass

    # TODO: Do we save anything for commputing the diaonal with Kronecker kernels?
    # @staticmethod
    # def diagonal(kernel: CombinationKernel, inputs: f64["N D"], params: dict) -> f64["N N"]:
    #     pass

    @staticmethod
    def gram_solve(
        kernel: CombinationKernel, inputs: f64["N D"], params: dict, array: f64["N"]
    ) -> f64["N N"]:

        # THIS IMPLIMENTATION IS NAIVE, WE CAN DO BETTER.
        # - Firstly, we can take the array inside the kronecker product of inverse matrices in (*)
        # - Secondly, we should be able to solve against something better than the identity matrix.

        gram_matrices = []

        # --- TODO: rewrite (--- to ---) in a more JAX-friendly way!
        for k, params in zip(kernel.kernel_set, params):

            # Filter grid based on active dims.
            _, indices = jnp.unique(
                inputs[..., k.active_dims], return_index=True, axis=0
            )

            gram_matrices.append(
                k.gram_solve(k, inputs[indices], params, I(inputs.shape[0]))
            )  # <- (*)
        # ---

        Kxx_inv = reduce(jnp.kron, gram_matrices)  # <- (*)

        return jnp.matmul(Kxx_inv, array)


# This just acts as a constructor for using the Kronecker backend.
# This mathematically makes sense for 'ProductKernel'  - a kronecker product is a special case of a product kernel.
@dataclass
class KroneckerKernel(ProductKernel, KroneckerKernelComputation):
    name: Optional[str] = "Kronecker kernel"

    def __post_int__(self):
        self.kernel_set = [KroneckerKernelComputation]


##########################################
# Euclidean kernels
##########################################
@dataclass(repr=False)
class RBF(Kernel, DenseKernelComputation):
    """The Radial Basis Function (RBF) kernel."""

    name: Optional[str] = "Radial basis function kernel"

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"], params: Dict
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\ell` and variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( \\frac{\\lVert x - y \\rVert^2_2}{2 \\ell^2} \\Bigg)

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (Dict): Parameter set for which the kernel should be evaluated on.

        Returns:
            Array: The value of :math:`k(x, y)`
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
class Matern12(Kernel, DenseKernelComputation):
    """The Matérn kernel with smoothness parameter fixed at 0.5."""

    name: Optional[str] = "Matern 1/2"

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"], params: Dict
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\ell` and variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( -\\frac{\\lvert x-y \\rvert}{\\ell}  \\Bigg)

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (Dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * jnp.exp(-0.5 * euclidean_distance(x, y))
        return K.squeeze()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


@dataclass(repr=False)
class Matern32(Kernel, DenseKernelComputation):
    """The Matérn kernel with smoothness parameter fixed at 1.5."""

    name: Optional[str] = "Matern 3/2"

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"], params: Dict
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with lengthscale parameter :math:`\ell` and variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{3}\\lvert x-y \\rvert}{\\ell}  \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{3}\\lvert x-y\\rvert}{\\ell} \\Bigg)

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (Dict): Parameter set for which the kernel should be evaluated on.

        Returns:
            Array: The value of :math:`k(x, y)`
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
class Matern52(Kernel, DenseKernelComputation):
    """The Matérn kernel with smoothness parameter fixed at 2.5."""

    name: Optional[str] = "Matern 5/2"

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"], params: Dict
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with lengthscale parameter :math:`\ell` and variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{5}\\lvert x-y \\rvert}{\\ell} + \\frac{5\\lvert x - y \\rvert^2}{3\\ell^2} \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{5}\\lvert x-y\\rvert}{\\ell} \\Bigg)

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (Dict): Parameter set for which the kernel should be evaluated on.

        Returns:
            Array: The value of :math:`k(x, y)`
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
class Polynomial(Kernel, DenseKernelComputation):
    """The Polynomial kernel with variable degree."""

    name: Optional[str] = "Polynomial"
    degree: int = 1

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)
        self.name = f"Polynomial Degree: {self.degree}"

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"], params: Dict
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with shift parameter :math:`\alpha` and variance :math:`\sigma` through

        .. math::
            k(x, y) = \\Big( \\alpha + \\sigma^2 xy \\Big)^{d}

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (Dict): Parameter set for which the kernel should be evaluated on.

        Returns:
            Array: The value of :math:`k(x, y)`
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
class White(Kernel, DiagonalKernelComputation):
    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)
        self._params = {"variance": jnp.array([1.0])}

    def __call__(self, x: f64["1 D"], y: f64["1 D"], params: dict) -> f64["1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \delta(x-y)

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.

        Returns:
            Array: The value of :math:`k(x, y)`
        """
        K = jnp.all(jnp.equal(x, y)) * params["variance"]  # <- pseudo code.
        return K.squeeze()


##########################################
# Graph kernels
##########################################
@dataclass
class _EigenKernel:
    laplacian: Float[Array, "N N"]


@dataclass
class GraphKernel(Kernel, _EigenKernel, DenseKernelComputation):
    name: Optional[str] = "Graph kernel"

    def __post_init__(self):
        self.ndims = 1
        evals, self.evecs = jnp.linalg.eigh(self.laplacian)
        self.evals = evals.reshape(-1, 1)
        self.num_vertex = self.laplacian.shape[0]

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"], params: Dict
    ) -> Float[Array, "1"]:
        """Evaluate the graph kernel on a pair of vertices v_i, v_j.

        Args:
            x (jnp.DeviceArray): Index of the ith vertex
            y (jnp.DeviceArray): Index of the jth vertex
            params (Dict): Parameter set for which the kernel should be evaluated on.

        Returns:
            Array: The value of k(v_i, v_j).
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
    """Compute the squared distance between a pair of inputs."""
    return jnp.sum((x - y) ** 2)


def euclidean_distance(
    x: Float[Array, "1 D"], y: Float[Array, "1 D"]
) -> Float[Array, "1"]:
    """Compute the l1 norm between a pair of inputs."""
    return jnp.sqrt(jnp.maximum(jnp.sum((x - y) ** 2), 1e-36))


__all__ = [
    "Kernel",
    "CombinationKernel",
    "SumKernel",
    "ProductKernel",
    "RBF" "Matern12",
    "Matern32",
    "Matern52",
    "Polynomial",
    "GraphKernel",
    "squared_distance",
    "euclidean_distance",
    "gram",
    "cross_covariance",
    "diagonal",
]
