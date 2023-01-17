import abc
from typing import Callable, Dict

import jax.numpy as jnp
from jax import vmap
from jaxlinop import (
    ConstantDiagonalLinearOperator,
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
