# Copyright 2022 The JaxLinOp Contributors. All Rights Reserved.
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


from beartype.typing import (
    Tuple,
    Type,
    Union,
)
import jax
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.linops.linear_operator import LinearOperator
from gpjax.typing import Array


def identity(
    n: int,
) -> "gpjax.linops.identity_linear_operator.IdentityLinearOperator":  # noqa: F821
    """Identity matrix.

    Args:
        n (int): Size of the identity matrix.

    Returns
    -------
        IdentityLinearOperator: Identity matrix of shape [n, n].
    """
    from gpjax.linops.identity_linear_operator import IdentityLinearOperator

    return IdentityLinearOperator(size=n)


def to_dense(obj: Union[Float[Array, "..."], LinearOperator]):
    """
    Ensure an object is a dense matrix.

    Args:
        obj (Union[Float[Array, "..."], LinearOperator]): Linear operator to convert.

    Returns
    -------
        Float[Array, "..."]: Dense matrix.
    """
    if isinstance(obj, jnp.ndarray):
        return obj
    elif isinstance(obj, LinearOperator):
        return obj.to_dense()
    else:
        raise TypeError(
            "object of class {} cannot be made into a Tensor".format(
                obj.__class__.__name__
            )
        )


def to_linear_operator(obj: Union[Float[Array, "..."], LinearOperator]):
    """
    Ensure an object is a linear operator.

    Args:
        obj (Union[Float[Array, "..."], LinearOperator]): Linear operator to convert.

    Returns
    -------
        LinearOperator: Linear operator.
    """
    if isinstance(obj, LinearOperator):
        return obj

    elif isinstance(obj, jnp.ndarray):
        from gpjax.linops.dense_linear_operator import DenseLinearOperator

        return DenseLinearOperator.from_dense(obj)
    else:
        raise TypeError(
            "object of class {} cannot be made into a Tensor".format(
                obj.__class__.__name__
            )
        )


def check_shapes_match(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> None:
    """Check shapes of two objects.

    Args:
        shape1 (Tuple[int, "..."]): Shape of the first object.
        shape2 (Tuple[int, "..."]): Shape of the second object.

    Raises
    ------
        ValueError: Shapes of the two objects do not match.
    """
    if shape1 != shape2:
        raise ValueError(
            f"`shape1` must have shape {shape1}, but `shape2` has shape {shape2}."
        )


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
    "identity",
    "to_dense",
]
