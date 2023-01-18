from typing import Callable, Dict

import jax.numpy as jnp
from jaxtyping import Array, Float
from .base import AbstractKernelComputation


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
