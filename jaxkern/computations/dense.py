from typing import Callable, Dict

from jax import vmap
from jaxtyping import Array, Float
from .base import AbstractKernelComputation


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
