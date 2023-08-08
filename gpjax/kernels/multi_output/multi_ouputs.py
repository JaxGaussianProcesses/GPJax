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

import abc
from dataclasses import dataclass
from functools import partial

from beartype.typing import (
    Callable,
    List,
    Optional,
    Union,
)
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
    ScalarInt,
)
from jax import lax



@dataclass
class AbstractMultiOutputKernel(AbstractKernel):
    r"""A base class for multi-output kernels."""
    
    num_outputs: ScalarInt = static_field(1)

    def get_output_idxs(self, x: Float[Array, "... D"]) -> Float[Array, "... 1"]:
        r"""Slice out the output indicies from the input matrix.

        Args:
            x (Float[Array, "... D"]): The matrix or vector that is to be sliced.

        Returns
        -------
            Float[Array, "... 1"]: The output indicies of the input matrix.
        """
        return x[...,-1:]

    def get_input_without_output_idxs(self, x: Float[Array, "... D"]) -> Float[Array, "... D - 1"]:
        r"""Slice out all but the output indicies from the input matrix.

        Args:
            x (Float[Array, "... D"]): The matrix or vector that is to be sliced.

        Returns
        -------
            Float[Array, "... Q"]: The input matrix with output indicies removed
        """
        return x[..., :-1] 


@dataclass
class IndependentMultiOutputKernel(AbstractMultiOutputKernel):
    r""" A kernel assuming that each output is independent and modelled with its own GP"""

    kernels: List[AbstractKernel] = None
    compute_engine: AbstractKernelComputation = static_field(BlockDiagonalKernelComputation()) # todo

    def __post_init__(self):
        # Add kernels to a list, flattening out instances of this class therein, as in GPFlow kernels.
        kernels_list: List[AbstractKernel] = []

        for kernel in self.kernels:
            if not isinstance(kernel, AbstractKernel):
                raise TypeError("can only combine Kernel instances")  # pragma: no cover

            if isinstance(kernel, self.__class__):
                kernels_list.extend(kernel.kernels)
            else:
                kernels_list.append(kernel)

        self.kernels = kernels_list

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # if x and y share the same output, then eval the kernel correspoding to the shared output
        # otherwise return 0

        x_output = self.get_output_idxs(x) # output idx for input x
        y_output = self.get_output_idxs(y) # output idx for input y
        x = self.get_input_without_output_idxs(x) 
        y = self.get_input_without_output_idxs(y)

        return lax.select(x_output == y_output, self.kernels[x_z](x,y), 0)


@dataclass
class CoregionalisationKernel(AbstractMultiOutputKernel):
    r"""A coregionalisation kernel. At the moment we only support intinsic coregionalisation, i.e.
    we are only using a single latent function to model the correlation between outputs."""

    kernel: AbstractKernel = None
    rank: ScalarInt = static_field(1)
    compute_engine: AbstractKernelComputation = static_field(KroneckerProductKernelComputation()) # todo

    def __post_init__(self):
            self.W = param_field(0.1 * jnp.ones([self.num_outputs, self.rank], dtype=tf.float64))
            self.kappa = param_field(jnp.ones([self.num_outputs], dtype=tf.float64), bijector=tfb.Softplus())
            self.B = jnp.matul(self.W, self.W.T) + jnp.diag(self.kappa)

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0.  todo what is z

        x_output = self.get_output_idxs(x) # output idx for input x
        y_output = self.get_output_idxs(y) # output idx for input y
        x = self.get_input_without_output_idxs(x) 
        y = self.get_input_without_output_idxs(y)

        return self.kernel(x,y) * self.B[x_output, y_output]







