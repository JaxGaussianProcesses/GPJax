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
    Array,
    Float,
    Int,
)
from simple_pytree import static_field
import tensorflow_probability.substrates.jax as tfp

from gpjax.base import param_field
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    EigenKernelComputation,
)
from gpjax.kernels.non_euclidean.utils import jax_gather_nd

tfb = tfp.bijectors


##########################################
# Graph kernels
##########################################
@dataclass
class GraphKernel(AbstractKernel):
    """The Matérn graph kernel defined on the vertex set of a graph.

    A Matérn graph kernel defined on the vertices of a graph. The key reference
    for this object is borovitskiy et. al., (2020).

    Args:
        laplacian (Float[Array]): An N x N matrix representing the Laplacian matrix
            of a graph.
        compute_engine
    """

    laplacian: Float[Array, "N N"] = static_field(None)
    lengthscale: Float[Array, " D"] = param_field(
        jnp.array([1.0]), bijector=tfb.Softplus()
    )
    variance: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    smoothness: Float[Array, "1"] = param_field(
        jnp.array([1.0]), bijector=tfb.Softplus()
    )
    eigenvalues: Float[Array, " N"] = static_field(None)
    eigenvectors: Float[Array, "N N"] = static_field(None)
    num_vertex: Int[Array, "1"] = static_field(None)
    compute_engine: AbstractKernelComputation = static_field(EigenKernelComputation)
    name: str = "Graph Matérn"

    def __post_init__(self):
        if self.laplacian is None:
            raise ValueError("Graph laplacian must be specified")

        evals, self.eigenvectors = jnp.linalg.eigh(self.laplacian)
        self.eigenvalues = evals.reshape(-1, 1)
        if self.num_vertex is None:
            self.num_vertex = self.eigenvalues.shape[0]

    def __call__(
        self,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
        **kwargs,
    ) -> Float[Array, "1"]:
        r"""Compute the (co)variance between a vertex pair.

        For a graph $\mathcal{G} = \{V, E\}$ where $V = \{v_1, v_2, \ldots v_n \},
        evaluate the graph kernel on a pair of vertices $(v_i, v_j)$ for any $i,j<n$.

        Args:
            x (Float[Array, "1 D"]): Index of the ith vertex.
            y (Float[Array, "1 D"]): Index of the jth vertex.

        Returns
        -------
            Float[Array, "1"]: The value of $k(v_i, v_j)$.
        """
        S = kwargs["S"]
        Kxx = (jax_gather_nd(self.eigenvectors, x) * S.squeeze()) @ jnp.transpose(
            jax_gather_nd(self.eigenvectors, y)
        )  # shape (n,n)
        return Kxx.squeeze()
