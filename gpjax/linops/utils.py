# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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

from numbers import Number

from beartype.typing import (
    Any,
    Type,
    Union,
)
import jax
from jax._src import dtypes
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.linops.base import AbstractLinearOperator
from gpjax.typing import Array


def to_dense(
    obj: Union[Float[Array, "..."], AbstractLinearOperator]
) -> Float[Array, "..."]:
    """
    Ensure an object is a dense matrix.

    Args:
        obj (Union[Float[Array, "..."], AbstractLinearOperator]): Linear operator to convert.

    Returns
    -------
        Float[Array, "..."]: Dense matrix.
    """
    if isinstance(obj, jax.Array):
        return obj
    elif isinstance(obj, AbstractLinearOperator):
        return obj.to_dense()
    else:
        raise TypeError(
            "object of class {} cannot be made into a Tensor".format(
                obj.__class__.__name__
            )
        )


def to_linear_operator(
    obj: Union[Float[Array, "..."], AbstractLinearOperator, Number]
) -> AbstractLinearOperator:
    """
    Ensure an object is a linear operator.

    Args:
        obj (Union[Float[Array, "..."], AbstractLinearOperator]): Linear operator to convert.

    Returns
    -------
        AbstractLinearOperator: Linear operator.
    """
    if isinstance(obj, AbstractLinearOperator):
        return obj

    elif isinstance(obj, jax.Array):
        from gpjax.linops.dense import Dense

        return Dense.from_dense(jnp.atleast_1d(obj))

    elif isinstance(obj, Number):
        from gpjax.linops.dense import Dense

        return Dense.from_dense(jnp.atleast_1d(jnp.array(obj)))

    else:
        raise TypeError(
            "object of class {} cannot be made into a Tensor".format(
                obj.__class__.__name__
            )
        )


def _check_same_shape(
    linop1: AbstractLinearOperator, linop2: AbstractLinearOperator
) -> None:
    """Check that two linear operators have the same shape.

    This is useful for checking that two linear operators are compatible for the add or mul arithmetic.

    Args:
        linop1 (AbstractLinearOperator): Linear operator.
        linop2 (AbstractLinearOperator): Linear operator.

    Raises
    ------
        ValueError: Shapes of the two objects do not match.
    """
    if linop1.shape != linop2:
        raise ValueError(
            f"`linops` must have same shape, but `linop1.shape = {linop1.shape}` and `linop2.shape = {linop2.shape}`."
        )


def _check_compatible_shape(
    linop1: AbstractLinearOperator, linop2: AbstractLinearOperator
) -> None:
    """Check that two linear operators have compatible shapes for multiplication."""

    if linop1.shape[1:] != linop2.shape[:-1]:
        raise ValueError(
            f"`linops` must have compatible shapes, but `linop1.shape = {linop1.shape}` and `linop2.shape = {linop2.shape}`."
        )


def _check_shape_arg(shape: Any) -> None:
    """Check shape argument.

    Args:
        shape (Any): Shape argument.

    Raises
    ------
        ValueError: Shape argument is not a tuple.
    """
    if not isinstance(shape, tuple):
        raise ValueError(f"`shape` must be a tuple, but `type(shape) = {type(shape)}`.")


def _check_dtype_arg(dtype: Any) -> None:
    """Check dtype argument.

    Args:
        dtype (Any): Dtype argument.

    Raises
    ------
        ValueError: Dtype argument is not a jax.numpy.dtype.
    """
    dtypes.check_user_dtype_supported(dtype, "array")


def default_dtype() -> Union[Type[jnp.float64], Type[jnp.float32]]:
    """Get the default dtype for the linear operator.

    Returns
    -------
        jnp.dtype: Default dtype for the linear operator.
    """
    if jax.config.x64_enabled:
        return jnp.float64
    else:
        return jnp.float32


__all__ = [
    "default_dtype",
    "to_linear_operator",
    "to_dense",
]
