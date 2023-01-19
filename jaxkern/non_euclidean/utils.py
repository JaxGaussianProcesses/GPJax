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

from jaxtyping import Num, Array, Int


def jax_gather_nd(
    params: Num[Array, "N ..."], indices: Int[Array, "M"]
) -> Num[Array, "M ..."]:
    """Slice a `params` array at a set of `indices`.

    Args:
        params (Num[Array]): An arbitrary array with leading axes of length `N` upon which we shall slice.
        indices (Float[Int]): An integer array of length M with values in the range [0, N) whose value at index `i` will be used to slice `params` at index `i`.

    Returns:
        Num[Array: An arbitrary array with leading axes of length `M`.
    """
    tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    return params[tuple_indices]
