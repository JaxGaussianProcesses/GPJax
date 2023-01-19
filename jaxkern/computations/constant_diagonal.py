from typing import Callable, Dict

from jax import vmap
from jaxlinop import (
    ConstantDiagonalLinearOperator,
    DiagonalLinearOperator,
)
from jaxtyping import Array, Float
from .base import AbstractKernelComputation


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
        return ConstantDiagonalLinearOperator(
            value=value.reshape(1), size=inputs.shape[0]
        )

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
