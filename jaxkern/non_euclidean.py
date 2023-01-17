from typing import Dict, List, Optional

import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array, Float

from .computations import EigenKernelComputation
from .nonstationary import AbstractKernel
from .utils import jax_gather_nd


##########################################
# Graph kernels
##########################################
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
        super().__init__(
            compute_engine, active_dims, stationary, spectral, name
        )
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

    def init_params(self, key: KeyArray) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "smoothness": jnp.array([1.0]),
        }

    @property
    def num_vertex(self) -> int:
        return self.compute_engine.num_vertex
