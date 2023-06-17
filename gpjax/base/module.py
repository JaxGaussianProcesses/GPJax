"""The base class for all differentiable GPJax modules."""
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
    r"""Declare a static field that is used to store metadata about the Module.

    Args:
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


class Module(Pytree):
    r"""The base class for all differentiable GPJax modules."""

    _pytree__meta: Dict[str, Any] = static_field()

    def __init_subclass__(cls, mutable: bool = False):
        r"""Register and set the attributes of the a subclass of `Module`.

        Args:
            mutable (bool, optional): Whether or not the subsclass should be mutable.
                Defaults to False.
        """
        cls._pytree__meta = {}
        super().__init_subclass__(mutable=mutable)
        class_vars = vars(cls)
        for field, value in class_vars.items():
            if (
                field not in cls._pytree__static_fields
                and isinstance(value, dataclasses.Field)
                and value.metadata is not None
            ):
                cls._pytree__meta[field] = {**value.metadata}

    def replace(self: Self, **kwargs: Any) -> Self:
        r"""
        Replace the values of the fields of the object.

        Args:
            **kwargs: keyword arguments to replace the fields of the object.

        Returns:
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
        r"""
        Replace the metadata of the fields.

        Args:
            **kwargs: keyword arguments to replace the metadata of the fields of the object.

        Returns:
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
        r"""
        Update the metadata of the fields. The metadata must already exist.

        Args:
            **kwargs: keyword arguments to replace the fields of the object.

        Returns:
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
        r"""Replace the trainability status of local nodes of the Module."""
        return self.update_meta(**{k: {"trainable": v} for k, v in kwargs.items()})

    def replace_bijector(self: Self, **kwargs: Dict[str, tfb.Bijector]) -> Self:
        r"""Replace the bijectors of local nodes of the Module."""
        return self.update_meta(**{k: {"bijector": v} for k, v in kwargs.items()})

    def constrain(self: Self) -> Self:
        r"""Transform model parameters to the constrained space according to their defined bijectors.

        Returns:
            Module: transformed to the constrained space.
        """

        def _apply_constrain(meta_leaf):
            """Apply the bijector to the leaf."""
            meta, leaf = meta_leaf

            if meta is None:
                return leaf

            return meta.get("bijector", tfb.Identity()).forward(leaf)

        return meta_map(_apply_constrain, self)

    def unconstrain(self: Self) -> Self:
        r"""Transform model parameters to the unconstrained space according to their
        defined bijectors.

        Returns:
            Module: transformed to the unconstrained space.
        """

        def _apply_unconstrain(meta_leaf):
            """Apply the inverse bijector to the leaf."""
            meta, leaf = meta_leaf

            if meta is None:
                return leaf

            return meta.get("bijector", tfb.Identity()).inverse(leaf)

        return meta_map(_apply_unconstrain, self)

    def stop_gradient(self: Self) -> Self:
        r"""Stop gradients flowing through the Module.

        Returns:
            Module: with gradients stopped.
        """

        def _stop_grad(leaf: jax.Array, trainable: bool) -> jax.Array:
            r"""Stop gradients flowing through a given leaf if it is not trainable."""
            return jax.lax.cond(trainable, lambda x: x, jax.lax.stop_gradient, leaf)

        def _apply_stop_grad(meta_leaf):
            """ "Apply the stop gradient function to the leaf."""
            meta, leaf = meta_leaf

            if meta is None:
                return leaf

            return _stop_grad(leaf, meta.get("trainable", True))

        return meta_map(_apply_stop_grad, self)

    def trainables(self: Self) -> Self:
        r"""Return a boolean mask of the trainability status of the Module.

        Returns:
            Self: A PyTree of booleans indicating the trainability status of the Module.
        """

        def _get_trainables(meta_leaf):
            """Get the trainability status of the leaf."""
            meta, leaf = meta_leaf
            if meta is None:
                return True

            return meta.get("trainable", True)

        return meta_map(_get_trainables, self)

    def __hash__(self):
        r"""Hash the Module by its leaves."""
        return hash(tuple(jtu.tree_leaves(self)))


def _toplevel_meta(pytree: Any) -> List[Optional[Dict[str, Any]]]:
    r"""Unpacks a list of meta corresponding to the top-level nodes of the pytree.

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
    r"""
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
        r"""Recursively unpack leaf metadata."""
        if is_leaf and is_leaf(pytree):
            yield meta_leaf
            return

        if type(pytree) in _registry:  # Registry tree trick, thanks to PyTreeClass!
            leaves_values, _ = _registry[type(pytree)].to_iter(pytree)
            leaves_meta = _toplevel_meta(pytree)

        elif pytree is not None:
            yield meta_leaf
            return

        for metadata, leaf in zip(leaves_meta, leaves_values):
            yield from _unpack_metadata((metadata, leaf), leaf, is_leaf)

    return list(_unpack_metadata(pytree, pytree, is_leaf))


def meta_flatten(
    pytree: Union[Module, Any], *, is_leaf: Optional[Callable[[Any], bool]] = None
) -> Union[Module, Any]:
    r"""
    Returns the meta of the Module.

    Args:
        pytree (Module): Module to get the meta of.
        is_leaf (Callable[[Any], bool]): predicate to determine if a node is a leaf.
            Defaults to None.

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
    r"""Apply a function to a Module where the first argument are the pytree leaves,
        and the second argument are the Module metadata leaves.
    Args:
        f (Callable[[Any, Dict[str, Any]], Any]): The function to apply to the pytree.
        pytree (Module): The pytree to apply the function to.
        rest (Any, optional): Additional pytrees to apply the function to. Defaults to None.
        is_leaf (Callable[[Any], bool], optional): predicate to determine if a node is
            a leaf. Defaults to None.

    Returns
    -------
        Module: The transformed pytree.
    """
    leaves, treedef = meta_flatten(pytree, is_leaf=is_leaf)
    all_leaves = [leaves] + [treedef.treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


def meta(pytree: Module, *, is_leaf: Optional[Callable[[Any], bool]] = None) -> Module:
    r"""Returns the metadata of the Module as a pytree.

    Args:
        pytree (Module): pytree to get the metadata of.

    Returns
    -------
        Module: metadata of the pytree.
    """

    def _filter_meta(meta_leaf):
        """Filter the metadata from the leaf."""
        meta, _ = meta_leaf
        return meta

    return meta_map(_filter_meta, pytree, is_leaf=is_leaf)


# Model saving and loading. Based upon the Flax checkpointing code
# https://github.com/google/flax/blob/main/flax/training/checkpoints.py
def _is_multiprocess_array(value: Any) -> bool:
    r"""Returns True if the value is a `multiprocess.Array`."""
    if isinstance(value, jax.Array):
        return not value.is_fully_addressable
    return False


def save_tree(
    path: str, model: Module, overwrite: bool = False, iterate: int = None
) -> None:
    r"""Save the `Module` to a given path.

    Args:
        path (str): The directory in which to save the model.
        model (Module): The model to save.
        overwrite (bool, optional): Whether or not to overwrite the model if it already
            exists. Defaults to False.
        iterate (int, optional): If the current optimisation iterate should be appended
            to the filename. Defaults to None.
    """

    def save_args_from_target(target: Any) -> Any:
        """Build the save args for a given leaf."""
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
    r"""Load a saved `Module` from a given path.

    Args:
        path (str): The path in which the model is saved.
        model (Module): The `Module` structure in which the loaded values should be
            placed.

    Returns:
        Module: A restored `Module` with the values loaded from the checkpoint.
    """

    def make_restore_args(x):
        """Build the restore args for a given leaf."""
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
