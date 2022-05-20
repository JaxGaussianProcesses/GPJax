import abc
from typing import Callable, Dict, List, Optional, Sequence

import jax.numpy as jnp
from chex import dataclass
from jax import vmap

from gpjax.types import Array


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
    _params: Optional[Dict] = None

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    @abc.abstractmethod
    def __call__(self, x: Array, y: Array, params: dict) -> Array:
        """Evaluate the kernel on a pair of inputs.
        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`.
        """
        raise NotImplementedError

    def slice_input(self, x: Array) -> Array:
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

    @property
    def ard(self):
        """Boolean property as to whether the kernel is isotropic or of automatic relevance determination form.

        Returns:
            bool: True if the kernel is an ARD kernel.
        """
        return True if self.ndims > 1 else False

    @property
    def params(self) -> dict:
        """A template dictionary of the kernel's parameter set."""
        return self._params

    @params.setter
    def params(self, value):
        """Override the kernel's initial parameter values.

        Args:
            value (dict): An alternative parameter dictionary that is to be used for initialisation.
        """
        self._params = value


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

    @property
    def params(self) -> List[Dict]:
        """A template dictionary of the kernel's parameter set."""
        return [kernel.params for kernel in self.kernel_set]

    def __call__(self, x: Array, y: Array, params: dict) -> Array:
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


##########################################
# Euclidean kernels
##########################################
@dataclass(repr=False)
class RBF(Kernel):
    """The Radial Basis Function (RBF) kernel."""

    name: Optional[str] = "Radial basis function kernel"

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)
        self._params = {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\ell` and variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( \\frac{\\lVert x - y \\rVert^2_2}{2 \\ell^2} \\Bigg)

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.

        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()


@dataclass(repr=False)
class Matern12(Kernel):
    """The Matérn kernel with smoothness parameter fixed at 0.5."""

    name: Optional[str] = "Matern 1/2"

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)
        self._params = {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\ell` and variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( -\\frac{\\lvert x-y \\rvert}{\\ell}  \\Bigg)

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * jnp.exp(-0.5 * euclidean_distance(x, y))
        return K.squeeze()


@dataclass(repr=False)
class Matern32(Kernel):
    """The Matérn kernel with smoothness parameter fixed at 1.5."""

    name: Optional[str] = "Matern 3/2"

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)
        self._params = {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with lengthscale parameter :math:`\ell` and variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{3}\\lvert x-y \\rvert}{\\ell}  \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{3}\\lvert x-y\\rvert}{\\ell} \\Bigg)

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.

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


@dataclass(repr=False)
class Matern52(Kernel):
    """The Matérn kernel with smoothness parameter fixed at 2.5."""

    name: Optional[str] = "Matern 5/2"

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)
        self._params = {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with lengthscale parameter :math:`\ell` and variance :math:`\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{5}\\lvert x-y \\rvert}{\\ell} + \\frac{5\\lvert x - y \\rvert^2}{3\\ell^2} \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{5}\\lvert x-y\\rvert}{\\ell} \\Bigg)

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.

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


@dataclass(repr=False)
class Polynomial(Kernel):
    """The Polynomial kernel with variable degree."""

    name: Optional[str] = "Polynomial"
    degree: int = 1

    def __post_init__(self):
        self.ndims = 1 if not self.active_dims else len(self.active_dims)
        self._params = {
            "shift": jnp.array([1.0]),
            "variance": jnp.array([1.0] * self.ndims),
        }
        self.name = f"Polynomial Degree: {self.degree}"

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with shift parameter :math:`\alpha` and variance :math:`\sigma` through

        .. math::
            k(x, y) = \\Big( \\alpha + \\sigma^2 xy \\Big)^{d}

        Args:
            x (jnp.DeviceArray): The left hand argument of the kernel function's call.
            y (jnp.DeviceArray): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.

        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x).squeeze()
        y = self.slice_input(y).squeeze()
        K = jnp.power(params["shift"] + jnp.dot(x * params["variance"], y), self.degree)
        return K.squeeze()


##########################################
# Graph kernels
##########################################
@dataclass
class _EigenKernel:
    laplacian: Array


@dataclass
class GraphKernel(Kernel, _EigenKernel):
    name: Optional[str] = "Graph kernel"

    def __post_init__(self):
        self.ndims = 1
        self._params = {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "smoothness": jnp.array([1.0]),
        }
        evals, self.evecs = jnp.linalg.eigh(self.laplacian)
        self.evals = evals.reshape(-1, 1)
        self.num_vertex = self.laplacian.shape[0]

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        """Evaluate the graph kernel on a pair of vertices v_i, v_j.

        Args:
            x (jnp.DeviceArray): Index of the ith vertex
            y (jnp.DeviceArray): Index of the jth vertex
            params (dict): Parameter set for which the kernel should be evaluated on.

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


def squared_distance(x: Array, y: Array):
    """Compute the squared distance between a pair of inputs."""
    return jnp.sum((x - y) ** 2)


def euclidean_distance(x: Array, y: Array):
    """Compute the l1 norm between a pair of inputs."""
    return jnp.sqrt(jnp.maximum(jnp.sum((x - y) ** 2), 1e-36))


def gram(kernel: Kernel, inputs: Array, params: dict) -> Array:
    """For a given kernel, compute the :math:`n \times n` gram matrix on an input matrix of shape :math:`n \times d` for :math:`d\geq 1`.

    Args:
        kernel (Kernel): The kernel for which the Gram matrix should be computed for.
        inputs (Array): The input matrix.
        params (dict): The kernel's parameter set.

    Returns:
        Array: The computed square Gram matrix.
    """
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(inputs))(inputs)


def cross_covariance(kernel: Kernel, x: Array, y: Array, params: dict) -> Array:
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


def diagonal(kernel: Kernel, inputs: Array, params: dict):
    """For a given kernel, compute the elementwise diagonal of the :math:`n \times n` gram matrix on an input matrix of shape :math:`n \times d` for :math:`d\geq 1`.
    Args:
        kernel (Kernel): The kernel for which the variance vector should be computed for.
        inputs (Array): The input matrix.
        params (dict): The kernel's parameter set.
    Returns:
        Array: The computed diagonal variance matrix.
    """
    return jnp.diag(vmap(lambda x: kernel(x, x, params))(inputs))
