import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import (
    Int,
)

from gpjax.kernels.base import AbstractKernel
from gpjax.parameters import (
    Parameter,
    PositiveReal,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


# Stationary kernels are a class of kernels that are invariant to translations in the input space.
class GraphEdgeKernel(AbstractKernel):
    r"""The Edge graph kernel defined on the edge set of a graph.

    Directed Graphs: K ((i, j), (i′, j′)) = 〈xi ⊗ xj, xi′ ⊗ xj′ 〉
    Undirected Graphs: K ((i, j), (i′, j′)) = 〈xi ⊗ xj, xi′ ⊗ xj′ 〉 + 〈xi ⊗ xj, xj′ ⊗ xi′ 〉

    """

    name: str = "Graph Matérn"

    def __init__(
        self,
        base_kernel,
        feature_mat,
        directed = False,
        active_dims: tp.Union[list[int], slice, None] = None,
        smoothness: ScalarFloat = 1.0,
        n_dims: tp.Union[int, None] = None,
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            smoothness: the smoothness parameter of the Matérn kernel.
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """
        if isinstance(smoothness, Parameter):
            self.smoothness = smoothness
        else:
            self.smoothness = PositiveReal(smoothness)

        self.base_kernel = base_kernel
        self.dense_feature_mat = feature_mat
        self.directed = directed

        super().__init__(active_dims, n_dims)

    def __call__(  # TODO not consistent with general kernel interface
        self,
        X: Int[Array, "N 2"],
        y: Int[Array, "N 2"] = None,
        *,
        S=None,
        **kwargs,
    ):
        """
        :param X: Specifies node indices for each edge in the batch. Shape
        [B, 2].
        :param X2: Specifies node indices for each edge in the batch. Shape
        [B', 2].
        :param presliced:
        :return:
            Kernel
        """

        X2 = X if y is None else y
        cov = self.base_kernel.gram(self.dense_feature_mat).to_dense()

        cov_edges = (jnp.take(jnp.take(cov, X[:, 0], axis=0), X2[:, 0], axis=1)
                     * jnp.take(jnp.take(cov, X[:, 1], axis=0), X2[:, 1], axis=1))

        if not self.directed:
            cov_edges += (jnp.take(jnp.take(cov, X[:, 0], axis=0), X2[:, 1], axis=1)
                        * jnp.take(jnp.take(cov, X[:, 1], axis=0), X2[:, 0], axis=1))

        return cov_edges.T.squeeze()
