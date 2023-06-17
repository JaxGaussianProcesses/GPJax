"""Parameter field for use in GPJax Modules."""
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


__all__ = ["param_field"]

import dataclasses

from beartype.typing import (
    Any,
    Mapping,
    Optional,
)
import tensorflow_probability.substrates.jax.bijectors as tfb


def param_field(  # noqa: PLR0913
    default: Any = dataclasses.MISSING,
    *,
    bijector: Optional[tfb.Bijector] = None,
    trainable: bool = True,
    default_factory: Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: Optional[bool] = None,
    compare: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
):
    r"""Define a dataclass field for a GPJax parameter.

    Fields defined in this way are fully compatible with primitive JAX operations
    such as `grad`. However, they come with additional functionality such as bijective
    transformations and trainable flags.

    Args:
        bijector (Optional[tfb.Bijector], optional): The bijector that is to be used
            to project the parameter's value to and from an unconstrained space. The
            field's value is `None` by default which leads to a `tfb.Identity` bijector
            being used. Defaults to None.
        trainable (bool, optional): Boolean field as to whether a gradient should be
            compute against the parameter's value when calling `jax.grad`. Defaults to
            True.
        default_factory (Any, optional):  If provided, it must be a zero-argument
            callable that will be called when a default value is needed for this field.
            Among other purposes, this can be used to specify fields with mutable
            default values. Defaults to dataclasses.MISSING.
        init (bool, optional): If True, this field is included as a parameter to the
            generated `__repr__()` method. Defaults to True.
        repr (bool, optional): If true, this field is included in the string returned
            by the generated . Defaults to True.
        hash (Optional[bool], optional): This can be a bool or None. If True, this
            field is included in the generated `__hash__()` method. If None, use the
            value of compare: this would normally be the expected behavior. A field
            should be considered in the hash if it's used for comparisons. Defaults to
            None.
        compare (bool, optional): If True, this field is included in the generated
            equality and comparison methods (`__eq__()`, `__gt__()`, et al.). Defaults
            to True.
        metadata (Optional[Mapping[str, Any]], optional): This can be a mapping or
            None. None is treated as an empty dict. Defaults to None.

    Returns:
        Field: The initialised field.
    """
    metadata = {} if metadata is None else dict(metadata)

    if "bijector" in metadata:
        raise ValueError("Cannot use metadata with `bijector` already set.")

    if "trainable" in metadata:
        raise ValueError("Cannot use metadata with `trainable` already set.")

    if "pytree_node" in metadata:
        raise ValueError("Cannot use metadata with `pytree_node` already set.")
    if bijector is None:
        bijector = tfb.Identity()
    metadata["bijector"] = bijector
    metadata["trainable"] = trainable
    metadata["pytree_node"] = True

    if (
        default is not dataclasses.MISSING
        and default_factory is not dataclasses.MISSING
    ):
        raise ValueError("Cannot specify both default and default_factory.")

    if default is not dataclasses.MISSING:
        default_factory = lambda: default

    return dataclasses.field(
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )
