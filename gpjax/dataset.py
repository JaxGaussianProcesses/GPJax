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

from dataclasses import dataclass

from beartype.typing import (
    Optional,
    Union,
)
import jax.numpy as jnp
from jaxtyping import (
    Bool,
    Num,
)
from simple_pytree import Pytree

from gpjax.typing import Array


class _Missing:
    """Sentinel class for not-yet-computed mask"""


@dataclass
class Dataset(Pytree):
    r"""Base class for datasets.

    Attributes
    ----------
        X: Num[Array, "N D"], optional:
            Input data.
        y: Num[Array, "N Q"], optional:
            Output data.
        mask: Bool[Array, "N Q"] or None, optional
            Mask for the output data. By default, a mask will be computed based on the output data.
            User can optionally specify a pre-computed mask, or pass `None` which means no mask
            will be computed.
    """

    X: Optional[Num[Array, "N D"]] = None
    y: Optional[Num[Array, "N Q"]] = None
    mask: Optional[Union[Bool[Array, "N Q"], None]] = _Missing()

    def __post_init__(self) -> None:
        r"""Checks that the shapes of $`X`$ and $`y`$ are compatible."""
        _check_shape(self.X, self.y)
        if isinstance(self.mask, _Missing):
            if self.y is not None:
                if jnp.any(mask := jnp.isnan(self.y)):
                    self.mask = mask
                else:
                    self.mask = None
            else:
                self.mask = None

    def __repr__(self) -> str:
        r"""Returns a string representation of the dataset."""
        repr = (
            f"- Number of observations: {self.n}\n- Input dimension:"
            f" {self.in_dim}\n- Output dimension: {self.out_dim}"
        )
        return repr

    def is_supervised(self) -> bool:
        r"""Returns `True` if the dataset is supervised."""
        return self.X is not None and self.y is not None

    def is_unsupervised(self) -> bool:
        r"""Returns `True` if the dataset is unsupervised."""
        return self.X is None and self.y is not None

    def __add__(self, other: "Dataset") -> "Dataset":
        r"""Combine two datasets. Right hand dataset is stacked beneath the left."""
        X = None
        y = None
        mask = None

        if self.X is not None and other.X is not None:
            X = jnp.concatenate((self.X, other.X))

        if self.y is not None and other.y is not None:
            y = jnp.concatenate((self.y, other.y))

        if (sm := self.mask is not None) | (om := other.mask is not None):
            sm = self.mask if sm else jnp.zeros(self.y.shape, dtype=bool)
            om = other.mask if om else jnp.zeros(other.y.shape, dtype=bool)
            mask = jnp.concatenate((sm, om))

        return Dataset(X=X, y=y, mask=mask)

    @property
    def n(self) -> int:
        r"""Number of observations."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        r"""Dimension of the inputs, $`X`$."""
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        r"""Dimension of the outputs, $`y`$."""
        return self.y.shape[1]


def _check_shape(
    X: Optional[Num[Array, "..."]], y: Optional[Num[Array, "..."]]
) -> None:
    r"""Checks that the shapes of $`X`$ and $`y`$ are compatible."""
    if X is not None and y is not None and X.shape[0] != y.shape[0]:
        raise ValueError(
            "Inputs, X, and outputs, y, must have the same number of rows."
            f" Got X.shape={X.shape} and y.shape={y.shape}."
        )

    if X is not None and X.ndim != 2:
        raise ValueError(
            f"Inputs, X, must be a 2-dimensional array. Got X.ndim={X.ndim}."
        )

    if y is not None and y.ndim != 2:
        raise ValueError(
            f"Outputs, y, must be a 2-dimensional array. Got y.ndim={y.ndim}."
        )


__all__ = [
    "Dataset",
]
