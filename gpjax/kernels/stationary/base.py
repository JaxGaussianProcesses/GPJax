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
from flax.experimental import nnx
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.parameters import PositiveReal
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class StationaryKernel(AbstractKernel):
    """Base class for stationary kernels."""

    def __init__(
        self,
        active_dims: tp.Union[list[int], int, slice],
        lengthscale: tp.Union[LengthscaleCompatible, nnx.Variable[Lengthscale]] = 1.0,
        variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        super().__init__(active_dims=active_dims, compute_engine=compute_engine)

        self.n_dims = _check_lengthscale_dims_compat(lengthscale, self.n_dims)

        if isinstance(lengthscale, nnx.Variable):
            self.lengthscale = lengthscale
        else:
            self.lengthscale = PositiveReal(lengthscale)

            # static typing
            if tp.TYPE_CHECKING:
                self.lengthscale = tp.cast(PositiveReal[Lengthscale], self.lengthscale)

        if isinstance(variance, nnx.Variable):
            self.variance = variance
        else:
            self.variance = PositiveReal(variance)

            # static typing
            if tp.TYPE_CHECKING:
                self.variance = tp.cast(PositiveReal[ScalarFloat], self.variance)


# TODO: maybe improve the control flow here
def _check_lengthscale_dims_compat(
    lengthscale: tp.Union[LengthscaleCompatible, nnx.Variable[Lengthscale]], n_dims: int
) -> tp.Union[int, None]:
    r"""Check that the lengthscale parameter is compatible with the number of input dimensions.

    Args:
        lengthscale (Float[Array, " D"]): The lengthscale parameter.
        n_dims (int): The number of input dimensions.
    """
    ls_shape = jnp.shape(lengthscale)

    if len(ls_shape) > 1:
        raise ValueError(
            "Expected `lengthscale` to be a scalar or 1D array. Got `lengthscale` with shape "
            f"{ls_shape}."
        )

    if ls_shape == ():
        return n_dims
    elif n_dims is None:
        return ls_shape[0]
    elif ls_shape != (n_dims,):
        raise ValueError(
            "Expected `lengthscale` to be compatible with the number "
            f"of input dimensions. Got `lengthscale` with shape {ls_shape}, "
            f"but the number of input dimensions is {n_dims}."
        )
