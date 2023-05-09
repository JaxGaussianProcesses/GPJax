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


from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import (
    Float,
    Int,
    Num,
)
import tensorflow_probability.substrates.jax as tfp

from gpjax.base import (
    param_field,
    static_field,
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

tfb = tfp.bijectors


##########################################
# Graph kernels
##########################################
@dataclass
class GraphKernel(AbstractKernel):
    r"""The Matérn graph kernel defined on the vertex set of a graph.

    A Matérn graph kernel defined on the vertices of a graph. The key reference
    for this object is borovitskiy et. al., (2020).

    Args:
        laplacian (Float[Array]): An $`N \times N`$ matrix representing the Laplacian matrix
            of a graph.
    """

    laplacian: Num[Array, "N N"] = static_field(None)
    lengthscale: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    smoothness: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    eigenvalues: Float[Array, " N"] = static_field(None)
    eigenvectors: Float[Array, "N N"] = static_field(None)
    num_vertex: ScalarInt = static_field(None)
    compute_engine: AbstractKernelComputation = static_field(EigenKernelComputation)
    name: str = "Graph Matérn"

    def __post_init__(self):
        if self.laplacian is None:
            raise ValueError("Graph laplacian must be specified")

        evals, self.eigenvectors = jnp.linalg.eigh(self.laplacian)
        self.eigenvalues = evals.reshape(-1, 1)
        if self.num_vertex is None:
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
