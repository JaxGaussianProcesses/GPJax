import beartype.typing as tp
from jaxtyping import (
    Int,
    Float
)
from typing import Optional
from gpjax.kernels.base import AbstractKernel

from gpjax.typing import (
    Array,
)

from gpjax.kernels.computations import (
    AbstractKernelComputation,
    GraphEdgeKernelComputation,
)

from gpjax.kernels.non_euclidean.utils import jax_gather


# Stationary kernels are a class of kernels that are invariant to translations in the input space.
class GraphEdgeKernel(AbstractKernel):
    r"""The Edge graph kernel defined on the edge set of a graph.
    The kernel is an implementation of Kai Yu, Wei Chu et. al.
    (https://papers.nips.cc/paper_files/paper/2007/hash/d045c59a90d7587d8d671b5f5aec4e7c-Abstract.html)

    Directed Graphs: K ((i, j), (i', j')) = 〈 xi ⊗ xj, xi' ⊗ xj' 〉
    Undirected Graphs: K ((i, j), (i', j')) = 〈 xi ⊗ xj, xi' ⊗ xj' 〉 + 〈xi ⊗ xj, xj' ⊗ xi's 〉
    Bipartite Graphs: K ((i, j), (i′, j′)) = 〈 xi ⊗ zj, xi′ ⊗ zj′ 〉

    """

    name: str = "Graph Matérn"

    def __init__(
        self,
        base_kernel,
        feature_mat,
        directed = False,
        active_dims: tp.Union[list[int], slice, None] = None,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = GraphEdgeKernelComputation(),
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

        self.base_kernel = base_kernel
        self.dense_feature_mat = feature_mat
        self.directed = directed

        super().__init__(active_dims, n_dims, compute_engine)

    def __call__(  # TODO not consistent with general kernel interface
        self,
        sender: Int[Array, "N 1"],
        reciever: Int[Array, "N 1"],
        sender_test: Optional[Int[Array, "N 1"]] = None,
        reciever_test: Optional[Int[Array, "N 1"]] = None,
        y: Float[Array, "N 2"] = None,
        *,
        S=None,
        **kwargs,
    ):
        """
        :param sender: Specifies the sending node indices for the edge in the batch. Shape
        [B, 1].
        :param reciever: Specifies the recieving node indices for each edge in the batch. Shape
        [B', 1].
        :param sender_z: Specifies the sending node indices for the edge in the batch when it is a bipartite graph. Shape
        [B, 1].
        :param reciever_z: Specifies the recieving node indices for each edge in the batch when it is a bipartite graph. Shape
        [B', 1].
        :return:
            Kernel
        """

        sender_test = sender_test if sender_test is not None else sender
        reciever_test = reciever_test if reciever_test is not None else reciever

        cov = self.base_kernel.gram(self.dense_feature_mat).to_dense()

        cov_edges = (jax_gather(jax_gather(cov, sender, axis=0), sender_test, axis=1)
                     * jax_gather(jax_gather(cov, reciever, axis=0), reciever_test, axis=1))

        if not self.directed:
            cov_edges += (jax_gather(jax_gather(cov, sender, axis=0), reciever_test, axis=1)
                        * jax_gather(jax_gather(cov, reciever, axis=0), sender_test, axis=1))

        return cov_edges.squeeze()
