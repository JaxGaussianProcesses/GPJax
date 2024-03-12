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

import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Int,
    Num,
)

from gpjax.kernels.computations import (
    AbstractKernelComputation,
    EigenKernelComputation,
)
from gpjax.kernels.non_euclidean.utils import jax_gather_nd
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.parameters import (
    Parameter,
    PositiveReal,
    Static,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
    ScalarInt,
)


class GraphKernel(StationaryKernel):
    r"""The Matérn graph kernel defined on the vertex set of a graph.

    A Matérn graph kernel defined on the vertices of a graph. The key reference
    for this object is borovitskiy et. al., (2020).

    Args:
        laplacian (Float[Array]): An $`N \times N`$ matrix representing the Laplacian matrix
            of a graph.
    """

    num_vertex: tp.Union[ScalarInt, None]
    laplacian: Static[Float[Array, "N N"]]
    eigenvalues: Static[Float[Array, "N 1"]]
    eigenvectors: Static[Float[Array, "N N"]]
    name: str = "Graph Matérn"

    def __init__(
        self,
        laplacian: Num[Array, "N N"],
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[ScalarFloat, Float[Array, " D"], Parameter] = 1.0,
        variance: tp.Union[ScalarFloat, Parameter] = 1.0,
        smoothness: ScalarFloat = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = EigenKernelComputation(),
    ):
        if isinstance(smoothness, Parameter):
            self.smoothness = smoothness
        else:
            self.smoothness = PositiveReal(smoothness)

        self.laplacian = Static(laplacian)
        evals, eigenvectors = jnp.linalg.eigh(self.laplacian.value)
        self.eigenvectors = Static(eigenvectors)
        self.eigenvalues = Static(evals.reshape(-1, 1))
        self.num_vertex = self.eigenvalues.value.shape[0]

        super().__init__(active_dims, lengthscale, variance, n_dims, compute_engine)

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
        Kxx = (jax_gather_nd(self.eigenvectors.value, x) * S.squeeze()) @ jnp.transpose(
            jax_gather_nd(self.eigenvectors.value, y)
        )  # shape (n,n)
        return Kxx.squeeze()
