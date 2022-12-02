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

import abc
import jax

from typing import Any


class Pytree(metaclass=abc.ABCMeta):
    """An abstract base class for a JAX compatible pytree. Adapted from `distrax._src.utils.jittable.Jittable`."""

    def __new__(cls, *args, **kwargs):
        # Discard the parameters to this function because the constructor is not
        # called during serialization: its `__dict__` gets repopulated directly.
        del args, kwargs
        try:
            registered_cls = jax.tree_util.register_pytree_node_class(cls)
        except ValueError:
            registered_cls = cls  # Already registered.
        return object.__new__(registered_cls)

    def tree_flatten(self):
        leaves, treedef = jax.tree_util.tree_flatten(self.__dict__)
        switch = list(map(is_jax_type, leaves))
        children = [leaf if s else None for leaf, s in zip(leaves, switch)]
        metadata = [None if s else leaf for leaf, s in zip(leaves, switch)]
        return children, (metadata, switch, treedef)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        metadata, switch, treedef = aux_data
        leaves = [j if s else p for j, p, s in zip(children, metadata, switch)]
        obj = object.__new__(cls)
        obj.__dict__ = jax.tree_util.tree_unflatten(treedef, leaves)
        return obj


def is_jax_type(x: Any) -> bool:
    """Check whether `x` is an instance of a JAX-compatible type."""
    # If it's a tracer, then it's already been converted by JAX.
    if isinstance(x, jax.core.Tracer):
        return True

    # `jax.vmap` replaces vmappable leaves with `object()` during serialization.
    if type(x) is object:  # pylint: disable=unidiomatic-typecheck
        return True

    # Primitive types (e.g. shape tuples) are treated as metadata for Distrax.
    if isinstance(x, (bool, int, float)) or x is None:
        return False

    # Otherwise, try to make it into a tracer. If it succeeds, then it's JAX data.
    try:
        jax.xla.abstractify(x)
        return True
    except TypeError:
        return False


__all__ = [
    "Pytree",
    "is_jax_type",
]
