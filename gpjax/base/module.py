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


__all__ = ["Module", "meta_leaves", "meta_flatten", "meta_map", "meta", "static_field"]

from copy import (
    copy,
    deepcopy,
)
import dataclasses
import os

from beartype.typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import jax
from jax._src.tree_util import _registry
import jax.tree_util as jtu
from orbax.checkpoint import (
    ArrayRestoreArgs,
    Checkpointer,
    PyTreeCheckpointHandler,
    RestoreArgs,
    SaveArgs,
)
from simple_pytree import Pytree
import tensorflow_probability.substrates.jax.bijectors as tfb

Self = TypeVar("Self")


def static_field(  # noqa: PLR0913
    default: Any = dataclasses.MISSING,
    *,
    default_factory: Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: Optional[bool] = None,
    compare: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
):
    metadata = {} if metadata is None else dict(metadata)

    if "pytree_node" in metadata:
        raise ValueError("Cannot use metadata with `pytree_node` already set.")

    metadata["pytree_node"] = False

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


def _inherited_metadata(cls: type) -> Dict[str]:
    meta_data = dict()
    for parent_class in cls.mro():
        if parent_class is not cls and parent_class is not Module:
            if issubclass(parent_class, Module):
                meta_data.update(parent_class._pytree__meta)
    return meta_data


class Module(Pytree):
    _pytree__meta: Dict[str, Any] = static_field()

    def __init_subclass__(cls, mutable: bool = False):
        super().__init_subclass__(mutable=mutable)
        cls._pytree__meta = _inherited_metadata(cls)
        class_vars = vars(cls)
        for field, value in class_vars.items():
            if (
                field not in cls._pytree__static_fields
                and isinstance(value, dataclasses.Field)
                and value.metadata is not None
            ):
                cls._pytree__meta[field] = {**value.metadata}

    def replace(self: Self, **kwargs: Any) -> Self:
        """
        Replace the values of the fields of the object.

        Args:
            **kwargs: keyword arguments to replace the fields of the object.

        Returns
        -------
            Module: with the fields replaced.
        """
        fields = vars(self)
        for key in kwargs:
            if key not in fields:
                raise ValueError(f"'{key}' is not a field of {type(self).__name__}")

        pytree = copy(self)
        pytree.__dict__.update(kwargs)
        return pytree

    def replace_meta(self: Self, **kwargs: Any) -> Self:
        """
        Replace the metadata of the fields.

        Args:
            **kwargs: keyword arguments to replace the metadata of the fields of the object.

        Returns
        -------
            Module: with the metadata of the fields replaced.
        """
        fields = vars(self)
        for key in kwargs:
            if key not in fields:
                raise ValueError(f"'{key}' is not a field of {type(self).__name__}")

        pytree = copy(self)
        pytree.__dict__.update(_pytree__meta={**pytree._pytree__meta, **kwargs})
        return pytree

    def update_meta(self: Self, **kwargs: Any) -> Self:
        """
        Update the metadata of the fields. The metadata must already exist.

        Args:
            **kwargs: keyword arguments to replace the fields of the object.

        Returns
        -------
            Module: with the fields replaced.
        """
        fields = vars(self)
        for key in kwargs:
            if key not in fields:
                raise ValueError(f"'{key}' is not a field of {type(self).__name__}")

        pytree = copy(self)
        new = deepcopy(pytree._pytree__meta)
        for key, value in kwargs.items():
            if key in new:
                new[key].update(value)
            else:
                new[key] = value
        pytree.__dict__.update(_pytree__meta=new)
        return pytree

    def replace_trainable(self: Self, **kwargs: Dict[str, bool]) -> Self:
        """Replace the trainability status of local nodes of the Module."""
        return self.update_meta(**{k: {"trainable": v} for k, v in kwargs.items()})

    def replace_bijector(self: Self, **kwargs: Dict[str, tfb.Bijector]) -> Self:
        """Replace the bijectors of local nodes of the Module."""
        return self.update_meta(**{k: {"bijector": v} for k, v in kwargs.items()})

    def constrain(self: Self) -> Self:
        """Transform model parameters to the constrained space according to their defined bijectors.

        Returns
        -------
            Module: transformed to the constrained space.
        """

        def _apply_constrain(meta_leaf):
            meta, leaf = meta_leaf

            if meta is None:
                return leaf

            return meta.get("bijector", tfb.Identity()).forward(leaf)

        return meta_map(_apply_constrain, self)

    def unconstrain(self: Self) -> Self:
        """Transform model parameters to the unconstrained space according to their defined bijectors.

        Returns
        -------
            Module: transformed to the unconstrained space.
        """

        def _apply_unconstrain(meta_leaf):
            meta, leaf = meta_leaf

            if meta is None:
                return leaf

            return meta.get("bijector", tfb.Identity()).inverse(leaf)

        return meta_map(_apply_unconstrain, self)

    def stop_gradient(self: Self) -> Self:
        """Stop gradients flowing through the Module.

        Returns
        -------
            Module: with gradients stopped.
        """

        # 🛑 Stop gradients flowing through a given leaf if it is not trainable.
        def _stop_grad(leaf: jax.Array, trainable: bool) -> jax.Array:
            return jax.lax.cond(trainable, lambda x: x, jax.lax.stop_gradient, leaf)

        def _apply_stop_grad(meta_leaf):
            meta, leaf = meta_leaf

            if meta is None:
                return leaf

            return _stop_grad(leaf, meta.get("trainable", True))

        return meta_map(_apply_stop_grad, self)

    def trainables(self: Self) -> Self:
        def _get_trainables(meta_leaf):
            meta, leaf = meta_leaf
            if meta is None:
                return True

            return meta.get("trainable", True)

        return meta_map(_get_trainables, self)


def _toplevel_meta(pytree: Any) -> List[Optional[Dict[str, Any]]]:
    """Unpacks a list of meta corresponding to the top-level nodes of the pytree.

    Args:
        pytree (Any): pytree to unpack the meta from.

    Returns
    -------
        List[Dict[str, Any]]: meta of the top-level nodes of the pytree.
    """
    if isinstance(pytree, Iterable):
        return [None] * len(pytree)
    return [
        pytree._pytree__meta.get(field, {})
        for field, _ in sorted(vars(pytree).items())
        if field not in pytree._pytree__static_fields
    ]


def meta_leaves(
    pytree: Module,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> List[Tuple[Optional[Dict[str, Any]], Any]]:
    """
    Returns the meta of the leaves of the pytree.

    Args:
        pytree (Module): pytree to get the meta of.
        is_leaf (Callable[[Any], bool]): predicate to determine if a node is a leaf. Defaults to None.

    Returns
    -------
        List[Tuple[Dict[str, Any], Any]]: meta of the leaves of the pytree.
    """

    def _unpack_metadata(
        meta_leaf: Any,
        pytree: Union[Module, Any],
        is_leaf: Optional[Callable[[Any], bool]],
    ):
        """Recursively unpack leaf metadata."""
        if is_leaf and is_leaf(pytree):
            yield meta_leaf
            return

        if type(pytree) in _registry:  # Registry tree trick, thanks to PyTreeClass!
            leaves_values, _ = _registry[type(pytree)].to_iter(pytree)
            leaves_meta = _toplevel_meta(pytree)

        elif pytree is not None:
            yield meta_leaf
            return

        for metadata, leaf in zip(leaves_meta, leaves_values, strict=True):
            yield from _unpack_metadata((metadata, leaf), leaf, is_leaf)

    return list(_unpack_metadata(pytree, pytree, is_leaf))


def meta_flatten(
    pytree: Union[Module, Any], *, is_leaf: Optional[Callable[[Any], bool]] = None
) -> Union[Module, Any]:
    """
    Returns the meta of the Module.

    Args:
        pytree (Module): Module to get the meta of.
        is_leaf (Callable[[Any], bool]): predicate to determine if a node is a leaf. Defaults to None.

    Returns
    -------
        Module: meta of the Module.
    """
    return meta_leaves(pytree, is_leaf=is_leaf), jtu.tree_structure(
        pytree, is_leaf=is_leaf
    )


def meta_map(
    f: Callable[[Any, Dict[str, Any]], Any],
    pytree: Union[Module, Any],
    *rest: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Union[Module, Any]:
    """Apply a function to a Module where the first argument are the pytree leaves, and the second argument are the Module metadata leaves.
    Args:
        f (Callable[[Any, Dict[str, Any]], Any]): The function to apply to the pytree.
        pytree (Module): The pytree to apply the function to.
        rest (Any, optional): Additional pytrees to apply the function to. Defaults to None.
        is_leaf (Callable[[Any], bool], optional): predicate to determine if a node is a leaf. Defaults to None.

    Returns
    -------
        Module: The transformed pytree.
    """
    leaves, treedef = meta_flatten(pytree, is_leaf=is_leaf)
    all_leaves = [leaves] + [treedef.treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves, strict=True))


def meta(pytree: Module, *, is_leaf: Optional[Callable[[Any], bool]] = None) -> Module:
    """Returns the metadata of the Module as a pytree.

    Args:
        pytree (Module): pytree to get the metadata of.

    Returns
    -------
        Module: metadata of the pytree.
    """

    def _filter_meta(meta_leaf):
        meta, _ = meta_leaf
        return meta

    return meta_map(_filter_meta, pytree, is_leaf=is_leaf)


# Model saving and loading. Based upon the Flax checkpointing code
# https://github.com/google/flax/blob/main/flax/training/checkpoints.py
def _is_multiprocess_array(value: Any) -> bool:
    if isinstance(value, jax.Array):
        return not value.is_fully_addressable
    return False


def save_tree(
    path: str, model: Module, overwrite: bool = False, iterate: int = None
) -> None:
    def save_args_from_target(target: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda x: SaveArgs(aggregate=not _is_multiprocess_array(x)), target
        )

    # Include the optimiser's iterate to the checkpoint path.
    if iterate:
        path = os.path.join(path, f"step_{iterate}")

    # Extract the leaves from the model.
    save_args = save_args_from_target(model)

    # Save the model.
    orbax_checkpointer = Checkpointer(PyTreeCheckpointHandler())
    orbax_checkpointer.save(path, model, save_args=save_args, force=overwrite)


def load_tree(path: str, model: Module) -> Module:
    def make_restore_args(x):
        if _is_multiprocess_array(x):
            return ArrayRestoreArgs(
                restore_type=jax.Array,
                sharding=x.sharding,
            )
        return RestoreArgs()

    restore_args = jax.tree_util.tree_map(make_restore_args, model)
    orbax_checkpointer = Checkpointer(PyTreeCheckpointHandler())
    restored = orbax_checkpointer.restore(path, item=model, restore_args=restore_args)
    return restored
