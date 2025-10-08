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

from jaxtyping import (
    Float,
    Int,
)

from gpjax.typing import Array
import jax.numpy as jnp
import functools
from jax import vmap


def jax_gather_nd(
    params: Float[Array, " N *rest"], indices: Int[Array, " M 1"]
) -> Float[Array, " M *rest"]:
    r"""Slice a `params` array at a set of `indices`.

    This is a reimplementation of TensorFlow's `gather_nd` function:
    [link](https://www.tensorflow.org/api_docs/python/tf/gather_nd)

    Args:
        params: an arbitrary array with leading axes of length $N$ upon
            which we shall slice.
        indices: an integer array of length $M$ with values in the range
            $[0, N)$ whose value at index $i$ will be used to slice `params` at
            index $i$.

    Returns:
        An arbitrary array with leading axes of length $M$.
    """
    tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    return params[tuple_indices]


def jax_gather(  # pylint: disable=unused-argument
    params: Float[Array, " N *rest"],
    indices: Int[Array, " M 1"],
    axis=None,
    batch_dims=0,
):
    r"""Gather slices from params axis `axis` according to indices.
    This is a reimplementation of TensorFlow's `gather_nd` function:
    [link](https://www.tensorflow.org/api_docs/python/tf/gather)
    Code is inspired from:
    [link](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/internal/backend/numpy/numpy_array.py#L84)


    Args:
    params: The `Tensor` from which to gather values. Must be at least rank
      `axis + 1`.
    indices: The index `Tensor`.  Must be one of the following types: `int32`,
      `int64`. The values must be in range `[0, params.shape[axis])`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`. The
      `axis` in `params` to gather `indices` from. Must be greater than or equal
      to `batch_dims`.  Defaults to the first non-batch dimension. Supports
      negative indexes.
    batch_dims: An `integer`.  The number of batch dimensions.  Must be less
      than or equal to `rank(indices)`.
    """

    if batch_dims < 0:
        raise NotImplementedError("Negative `batch_dims` is currently unsupported.")
    if axis is None:
        axis = batch_dims
    if axis < 0:
        axis = axis + len(params.shape)

    if batch_dims == 0 and axis == 0:
        return params[indices]
    take = lambda params, indices: jnp.take(
        params,
        indices,  # pylint: disable=g-long-lambda
        axis=axis - batch_dims,
    )
    take = functools.reduce(lambda g, f: f(g), [vmap] * int(batch_dims), take)
    return take(params, indices)
