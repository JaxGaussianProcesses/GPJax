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

import typing as tp

from cola.annotations import PSD
from cola.ops.operators import (
    Diagonal,
    Identity,
    Product,
)
from jax import vmap
import jax.numpy as jnp
from jaxtyping import Float

import gpjax
from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.typing import Array

K = tp.TypeVar("K", bound="gpjax.kernels.base.AbstractKernel")  # noqa: F821
ConstantDiagonalType = Product


class ConstantDiagonalKernelComputation(AbstractKernelComputation):
    r"""Computation engine for constant diagonal kernels."""

    def gram(self, kernel: K, x: Float[Array, "N D"]) -> Product:
        value = kernel(x[0], x[0])
        dtype = value.dtype
        shape = (x.shape[0], x.shape[0])
        return PSD(jnp.atleast_1d(value) * Identity(shape=shape, dtype=dtype))

    def _diagonal(self, kernel: K, inputs: Float[Array, "N D"]) -> Diagonal:
        diag = vmap(lambda x: kernel(x, x))(inputs)
        return PSD(Diagonal(diag=diag))

    def _cross_covariance(
        self, kernel: K, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        # TODO: This is currently a dense implementation. We should implement
        # a sparse LinearOperator for non-square cross-covariance matrices.
        cross_cov = vmap(lambda x: vmap(lambda y: kernel(x, y))(y))(x)
        return cross_cov
