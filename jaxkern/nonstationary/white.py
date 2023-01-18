from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..base import AbstractKernel
from ..computations import (
    ConstantDiagonalKernelComputation,
)


class White(AbstractKernel, ConstantDiagonalKernelComputation):
    def __post_init__(self) -> None:
        super(White, self).__post_init__()

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\delta(x-y)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        K = jnp.all(jnp.equal(x, y)) * params["variance"]
        return K.squeeze()

    def init_params(self, key: Float[Array, "1 D"]) -> Dict:
        """Initialise the kernel parameters.

        Args:
            key (Float[Array, "1 D"]): The key to initialise the parameters with.

        Returns:
            Dict: The initialised parameters.
        """
        return {"variance": jnp.array([1.0])}
