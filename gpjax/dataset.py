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
import warnings

from beartype.typing import (
    Literal,
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


@dataclass
class Dataset(Pytree):
    r"""Base class for datasets.

    Attributes
    ----------
        X (Optional[Num[Array, "N D"]]): input data.
        y (Optional[Num[Array, "N Q"]]): output data.
        mask (Optional[Union[Bool[Array, "N Q"], Literal["infer automatically"]]]): mask for the output data.
            Users can optionally specify a pre-computed mask, or explicitly pass `None` which
            means no mask will be used. Defaults to `"infer automatically"`, which means that
            the mask will be computed from the output data, or set to `None` if no output data is provided.
    """

    X: Optional[Num[Array, "N D"]] = None
    y: Optional[Num[Array, "N Q"]] = None
    mask: Optional[
        Union[Bool[Array, "N Q"], Literal["infer automatically"]]
    ] = "infer automatically"

    def __post_init__(self) -> None:
        r"""Checks that the shapes of $`X`$ and $`y`$ are compatible,
        and provides warnings regarding the precision of $`X`$ and $`y`$."""
        _check_shape(self.X, self.y)
        _check_precision(self.X, self.y)

        if isinstance(self.mask, str):
            if not self.mask == "infer automatically":
                raise ValueError(
                    f"mask must be either the string 'infer automatically', None, or a boolean array."
                    f" Got mask={self.mask}."
                )
            elif self.y is not None:
                mask = jnp.isnan(self.y)
                if jnp.any(mask):
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

        self_m_exists = self.mask is not None
        other_m_exists = other.mask is not None
        self_m = self.mask if self_m_exists else jnp.zeros(self.y.shape, dtype=bool)
        other_m = other.mask if other_m_exists else jnp.zeros(other.y.shape, dtype=bool)
        if self_m_exists or other_m_exists:
            mask = jnp.concatenate((self_m, other_m))

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


def _check_precision(
    X: Optional[Num[Array, "..."]], y: Optional[Num[Array, "..."]]
) -> None:
    r"""Checks the precision of $`X`$ and $`y`."""
    if X is not None and X.dtype != jnp.float64:
        warnings.warn(
            "X is not of type float64. "
            f"Got X.dtype={X.dtype}. This may lead to numerical instability. ",
            stacklevel=2,
        )

    if y is not None and y.dtype != jnp.float64:
        warnings.warn(
            "y is not of type float64."
            f"Got y.dtype={y.dtype}. This may lead to numerical instability.",
            stacklevel=2,
        )


__all__ = [
    "Dataset",
]
