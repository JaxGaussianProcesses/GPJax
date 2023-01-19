from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array

from ..base import AbstractKernel
from ..computations import (
    DenseKernelComputation,
)


##########################################
# Euclidean kernels
##########################################
class Linear(AbstractKernel):
    """The linear kernel."""

    def __init__(
        self,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        name: Optional[str] = "Linear",
    ) -> None:
        super().__init__(DenseKernelComputation, active_dims, stationary, None, name)

    def __call__(self, params: dict, x: jax.Array, y: jax.Array) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance parameter :math:`\\sigma`

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

    def init_params(self, key: KeyArray) -> Dict:
        return {"variance": jnp.array([1.0])}
