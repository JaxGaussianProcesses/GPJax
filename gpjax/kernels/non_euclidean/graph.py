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
import tensorflow_probability.substrates.jax as tfp
from gpjax.typing import Array
from jaxtyping import Float, Num, Int
from simple_pytree import static_field
from gpjax.typing import ScalarFloat

from ...base import param_field
from ..base import AbstractKernel
from ..computations import AbstractKernelComputation, EigenKernelComputation
from .utils import jax_gather_nd

tfb = tfp.bijectors


##########################################
# Graph kernels
##########################################
@dataclass
class GraphKernel(AbstractKernel):
    """A Matérn graph kernel defined on the vertices of a graph. The key reference for this object is borovitskiy et. al., (2020).

    Args:
        laplacian (Float[Array]): An N x N matrix representing the Laplacian matrix of a graph.
        compute_engine
    """

    laplacian: Float[Array, "N N"] = static_field(None)
    lengthscale: Float[Array, "D"] = param_field(
        jnp.array([1.0]), bijector=tfb.Softplus()
    )
    variance: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    smoothness: Float[Array, "1"] = param_field(
        jnp.array([1.0]), bijector=tfb.Softplus()
    )
    eigenvalues: Float[Array, "N"] = static_field(None)
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
        x: Float[Array, "D"],
        y: Float[Array, "D"],
        **kwargs,
    ) -> ScalarFloat:
        """Evaluate the graph kernel on a pair of vertices :math:`v_i, v_j`.

        Args:
            x (Float[Array, "D"]): Index of the ith vertex.
            y (Float[Array, "D"]): Index of the jth vertex.

        Returns:
            ScalarFloat: The value of :math:`k(v_i, v_j)`.
        """
        S = kwargs["S"]
        Kxx = (jax_gather_nd(self.eigenvectors, x) * S.squeeze()) @ jnp.transpose(
            jax_gather_nd(self.eigenvectors, y)
        )  # shape (n,n)
        return Kxx.squeeze()
