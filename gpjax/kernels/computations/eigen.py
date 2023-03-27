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

from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import AbstractKernelComputation
from dataclasses import dataclass


@dataclass
class EigenKernelComputation(AbstractKernelComputation):
    eigenvalues: Float[Array, "N"] = None
    eigenvectors: Float[Array, "N N"] = None
    num_verticies: int = None

    def cross_covariance(
        self, params: Dict, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        # Extract the graph Laplacian's eigenvalues
        evals = self.eigenvalues
        # Transform the eigenvalues of the graph Laplacian according to the
        # RBF kernel's SPDE form.
        S = jnp.power(
            evals
            + 2
            * self.kernel.smoothness
            / self.kernel.lengthscale
            / self.kernel.lengthscale,
            -self.kernel.smoothness,
        )
        S = jnp.multiply(S, self.num_vertex / jnp.sum(S))
        # Scale the transform eigenvalues by the kernel variance
        S = jnp.multiply(S, params["variance"])
        return self.kernel(x, y, S=S)
