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

from typing import Dict, List, Optional

import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array, Float
from dataclasses import dataclass
from ..computations import EigenKernelComputation
from ..base import AbstractKernel
from ...parameters import param_field
from .utils import jax_gather_nd
import tensorflow_probability.substrates.jax as tfp
tfb = tfp.bijectors

##########################################
# Graph kernels
##########################################
@dataclass
class AbstractGraphKernel:
    laplacian: Float[Array, "N N"]


@dataclass
class GraphKernel(AbstractKernel, AbstractGraphKernel):
    """A Matérn graph kernel defined on the vertices of a graph. The key reference for this object is borovitskiy et. al., (2020).

    Args:
        laplacian (Float[Array]): An N x N matrix representing the Laplacian matrix of a graph.
        compute_engine
    """
    lengthscale: Float[Array, "D"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus)
    variance: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus)
    smoothness: Float[Array, "1"] = param_field(jnp.array([1.0]), bijector=tfb.Softplus)

    def __post_init__(self):
        evals, self.evecs = jnp.linalg.eigh(self.laplacian)
        self.evals = evals.reshape(-1, 1)
        self.compute_engine.eigensystem = self.evals, self.evecs
        self.compute_engine.num_vertex = self.laplacian.shape[0]

    def __call__(
        self,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
        **kwargs,
    ) -> Float[Array, "1"]:
        """Evaluate the graph kernel on a pair of vertices :math:`v_i, v_j`.

        Args:
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

    @property
    def num_vertex(self) -> int:
        """The number of vertices within the graph.

        Returns:
            int: An integer representing the number of vertices within the graph.
        """
        return self.compute_engine.num_vertex
