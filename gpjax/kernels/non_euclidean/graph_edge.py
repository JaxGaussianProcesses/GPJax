import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import (
    Int,
    Num,
)

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    GraphEdgeKernelComputation,
)
from gpjax.typing import (
    Array,
)


# Stationary kernels are a class of kernels that are invariant to translations in the input space.
class GraphEdgeKernel(AbstractKernel):
    r"""The Edge graph kernel defined on the edge set of a graph.
    The kernel is an implementation of Kai Yu et al 2008
    https://papers.nips.cc/paper_files/paper/2007/hash/d045c59a90d7587d8d671b5f5aec4e7c-Abstract.html

    Directed Graphs: K ((i, j), (i', j')) = 〈 xi ⊗ xj, xi' ⊗ xj' 〉
    Undirected Graphs: K ((i, j), (i', j')) = 〈 xi ⊗ xj, xi' ⊗ xj' 〉 + 〈xi ⊗ xj, xj' ⊗ xi's 〉
    Bipartite Graphs: K ((i, j), (i′, j′)) = 〈 xi ⊗ zj, xi′ ⊗ zj′ 〉

    """

    name: str = "Graph Matérn"

    def __init__(
        self,
        base_kernel: AbstractKernel,
        feature_mat: Num[Array, "N M"],
        directed=False,
        active_dims: tp.Union[list[int], slice, None] = None,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = GraphEdgeKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            base_kernel: the node feature matrix of size (number of nodes, number of features)
            directed: True or false for directionality of graph edges
            active_dims: The indices of the input dimensions that the kernel operates on.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """

        self.base_kernel = base_kernel
        self.dense_feature_mat = feature_mat
        self.directed = directed

        super().__init__(active_dims, n_dims, compute_engine)

    def __call__(  # TODO not consistent with general kernel interface
        self,
        X: Int[Array, "N 2"],
        y: Int[Array, "M 2"],
        *,
        S=None,
        **kwargs,
    ):
        r"""
        :param sender: Specifies the sending node indices for the edge in the batch. Shape
        [B, 2].
        :param reciever: Specifies the recieving node indices for each edge in the batch. Shape
        [B', 2].
        :return:
            Kernel
        """

        sender, reciever = X[:, 0], X[:, 1]
        sender_test, reciever_test = y[:, 0], y[:, 1]

        cov = self.base_kernel.gram(self.dense_feature_mat).to_dense()

        cov_edges = jnp.take(
            jnp.take(cov, sender, axis=0), sender_test, axis=1
        ) * jnp.take(jnp.take(cov, reciever, axis=0), reciever_test, axis=1)

        if not self.directed:
            cov_edges += jnp.take(
                jnp.take(cov, sender, axis=0), reciever_test, axis=1
            ) * jnp.take(jnp.take(cov, reciever, axis=0), sender_test, axis=1)

        return cov_edges.squeeze()
