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

from __future__ import annotations

__all__ = ["param_field"]

import dataclasses
from beartype.typing import Any, Mapping, Optional

import tensorflow_probability.substrates.jax.bijectors as tfb


def param_field(
    default: Any = dataclasses.MISSING,
    *,
    bijector: tfb.Bijector = tfb.Identity(),
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
