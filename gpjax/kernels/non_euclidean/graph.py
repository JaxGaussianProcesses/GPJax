# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import field

from flax.experimental import nnx
import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Int,
    Num,
)

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    EigenKernelComputation,
)
from gpjax.kernels.non_euclidean.utils import jax_gather_nd
from gpjax.typing import (
    Array,
    ScalarFloat,
    ScalarInt,
)


class GraphKernel(AbstractKernel):
    r"""The Matérn graph kernel defined on the vertex set of a graph.

    A Matérn graph kernel defined on the vertices of a graph. The key reference
    for this object is borovitskiy et. al., (2020).

    Args:
        laplacian (Float[Array]): An $`N \times N`$ matrix representing the Laplacian matrix
            of a graph.
    """

    num_vertex: tp.Union[ScalarInt, None]
    eigenvalues: Float[Array, "N 1"]
    eigenvectors: Float[Array, "N N"]
    name: str = "Graph Matérn"

    def __init__(
        self,
        laplacian: Num[Array, "N N"],
        active_dims: tp.Union[list[int], int, slice],
        lengthscale: tp.Union[ScalarFloat, Float[Array, " D"]] = 1.0,
        variance: ScalarFloat = 1.0,
        smoothness: ScalarFloat = 1.0,
        compute_engine: AbstractKernelComputation = EigenKernelComputation(),
    ):
        super().__init__(active_dims=active_dims, compute_engine=compute_engine)

        self.laplacian = laplacian
        self.lengthscale = lengthscale
        self.variance = variance
        self.smoothness = smoothness

        evals, self.eigenvectors = jnp.linalg.eigh(self.laplacian)
        self.eigenvalues = evals.reshape(-1, 1)
        self.num_vertex = self.eigenvalues.shape[0]

    def __call__(  # TODO not consistent with general kernel interface
        self,
        x: Int[Array, "N 1"],
        y: Int[Array, "N 1"],
        *,
        S,
        **kwargs,
    ):
        r"""Compute the (co)variance between a vertex pair.

        For a graph $`\mathcal{G} = \{V, E\}`$ where $`V = \{v_1, v_2, \ldots v_n \}`$,
        evaluate the graph kernel on a pair of vertices $`(v_i, v_j)`$ for any $`i,j<n`$.

        Args:
            x (Float[Array, "N 1"]): Index of the $`i`$th vertex.
            y (Float[Array, "N 1"]): Index of the $`j`$th vertex.

        Returns
        -------
            ScalarFloat: The value of $k(v_i, v_j)$.
        """
        Kxx = (jax_gather_nd(self.eigenvectors, x) * S.squeeze()) @ jnp.transpose(
            jax_gather_nd(self.eigenvectors, y)
        )  # shape (n,n)
        return Kxx.squeeze()
