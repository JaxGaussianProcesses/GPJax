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

import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Optional
from simple_pytree import Pytree
from dataclasses import dataclass


@dataclass
class Dataset(Pytree):
    """Base class for datasets.

    Attributes:
        X (Optional[Float[Array, "N D"]]): Input data.
        y (Optional[Float[Array, "N Q"]]): Output data.
    """

    X: Optional[Float[Array, "N D"]] = None
    y: Optional[Float[Array, "N Q"]] = None

    def __post_init__(self) -> None:
        """Checks that the shapes of X and y are compatible."""
        _check_shape(self.X, self.y)

    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        repr = (
            f"- Number of observations: {self.n}\n- Input dimension:"
            f" {self.in_dim}\n- Output dimension: {self.out_dim}"
        )
        return repr

    def is_supervised(self) -> bool:
        """Returns `True` if the dataset is supervised."""
        return self.X is not None and self.y is not None

    def is_unsupervised(self) -> bool:
        """Returns `True` if the dataset is unsupervised."""
        return self.X is None and self.y is not None

    def __add__(self, other: Dataset) -> Dataset:
        """Combine two datasets. Right hand dataset is stacked beneath the left."""
        X = jnp.concatenate((self.X, other.X))
        y = jnp.concatenate((self.y, other.y))

        return Dataset(X=X, y=y)

    @property
    def n(self) -> int:
        """Number of observations."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        """Dimension of the inputs, X."""
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        """Dimension of the outputs, y."""
        return self.y.shape[1]


def _check_shape(X: Float[Array, "N D"], y: Float[Array, "N Q"]) -> None:
    """Checks that the shapes of X and y are compatible."""
    if X is not None and y is not None:
        if X.shape[0] != y.shape[0]:
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
