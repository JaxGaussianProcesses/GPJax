from __future__ import annotations

__all__ = ["param_field"]

import dataclasses
from typing import Any, Mapping, Optional

from .bijectors import Bijector, Identity


def param_field(
    default: Any = dataclasses.MISSING,
    *,
    bijector: Bijector = Identity,
    trainable: bool = True,
    default_factory: Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: Optional[bool] = None,
    compare: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
):
    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)

    if "bijector" in metadata:
        raise ValueError("Cannot use metadata with `bijector` already set.")

    if "trainable" in metadata:
        raise ValueError("Cannot use metadata with `trainable` already set.")

    if "pytree_node" in metadata:
        raise ValueError("Cannot use metadata with `pytree_node` already set.")

    metadata["bijector"] = bijector
    metadata["trainable"] = trainable
    metadata["pytree_node"] = True

    return dataclasses.field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )
