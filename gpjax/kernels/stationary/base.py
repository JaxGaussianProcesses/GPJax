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
from jaxtyping import Float

from gpjax.parameters import Parameter, PositiveReal
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import AbstractKernelComputation, DenseKernelComputation
from gpjax.kernels.stationary.utils import (
    _check_lengthscale_dims_compat,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class StationaryKernel(AbstractKernel):
    """Base class for stationary kernels."""

    def __init__(
        self,
        active_dims: tp.Union[list[int], int, slice],
        lengthscale: tp.Union[ScalarFloat, Float[Array, " D"], Parameter] = 1.0,
        variance: tp.Union[ScalarFloat, Parameter] = 1.0,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        super().__init__(active_dims=active_dims, compute_engine=compute_engine)

        _check_lengthscale_dims_compat(lengthscale, self.n_dims)

        if isinstance(lengthscale, Parameter):
            self.lengthscale = lengthscale
        else:
            self.lengthscale = PositiveReal(lengthscale)

        if isinstance(variance, Parameter):
            self.variance = variance
        else:
            self.variance = PositiveReal(variance)
