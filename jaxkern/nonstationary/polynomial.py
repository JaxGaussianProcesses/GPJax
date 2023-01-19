from typing import Dict, List, Optional

import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array, Float

from ..base import AbstractKernel
from ..computations import (
    DenseKernelComputation,
)


class Polynomial(AbstractKernel):
    """The Polynomial kernel with variable degree."""

    def __init__(
        self,
        degree: int = 1,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        name: Optional[str] = "Polynomial",
    ) -> None:
        super().__init__(DenseKernelComputation, active_dims, stationary, None, name)
        self.degree = degree
        self.name = f"Polynomial Degree: {self.degree}"

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with shift parameter :math:`\\alpha` and variance :math:`\\sigma^2` through

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

    def init_params(self, key: KeyArray) -> Dict:
        return {
            "shift": jnp.array([1.0]),
            "variance": jnp.array([1.0] * self.ndims),
        }
