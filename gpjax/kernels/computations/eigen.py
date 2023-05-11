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
    Num,
)

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.typing import Array


@dataclass
class EigenKernelComputation(AbstractKernelComputation):
    r"""Eigen kernel computation class. Kernels who operate on an
    eigen-decomposed structure should use this computation object.
    """

    def cross_covariance(
        self, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "N M"]:
        r"""Compute the cross-covariance matrix.

        For an $`N\times D`$ and $`M\times D`$ pair of matrices, evaluate the $`N \times M`$
        cross-covariance matrix.

        Args:
            x (Float[Array,"N D"]): The input matrix.
            y (Float[Array,"M D"]): The input matrix.

        Returns:
            _type_: _description_
        """
        # Transform the eigenvalues of the graph Laplacian according to the
        # RBF kernel's SPDE form.
        S = jnp.power(
            self.kernel.eigenvalues
            + 2
            * self.kernel.smoothness
            / self.kernel.lengthscale
            / self.kernel.lengthscale,
            -self.kernel.smoothness,
        )
        S = jnp.multiply(S, self.kernel.num_vertex / jnp.sum(S))
        # Scale the transform eigenvalues by the kernel variance
        S = jnp.multiply(S, self.kernel.variance)
        return self.kernel(x, y, S=S)
