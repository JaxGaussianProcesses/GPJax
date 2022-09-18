# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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

import jax.numpy as jnp
from chex import dataclass
from jaxtyping import Array, Float

NoneType = type(None)
from chex import PRNGKey as PRNGKeyType


@dataclass
class Dataset:
    """GPJax Dataset class."""

    X: Float[Array, "N D"]
    y: Float[Array, "N 1"] = None

    def __repr__(self) -> str:
        return (
            f"- Number of datapoints: {self.X.shape[0]}\n- Dimension: {self.X.shape[1]}"
        )

    def __add__(self, other: "Dataset") -> "Dataset":
        """Combines two datasets into one. The right-hand dataset is stacked beneath left."""
        x = jnp.concatenate((self.X, other.X))
        y = jnp.concatenate((self.y, other.y))

        return Dataset(X=x, y=y)

    @property
    def n(self) -> int:
        """The number of observations in the dataset."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        """The dimension of the input data."""
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        """The dimension of the output data."""
        return self.y.shape[1]


def verify_dataset(ds: Dataset) -> None:
    """Apply a series of checks to the dataset to ensure that downstream operations are safe."""
    assert (
        ds.X.ndim == 2
    ), f"2-dimensional training inputs are required. Current dimension: {ds.X.ndim}."
    if ds.y is not None:
        assert (
            ds.y.ndim == 2
        ), f"2-dimensional training outputs are required. Current dimension: {ds.y.ndim}."
        assert ds.X.shape[0] == ds.y.shape[0], (
            "Number of inputs must equal the number of outputs. \nCurrent"
            f" counts:\n- X: {ds.X.shape[0]}\n- y: {ds.y.shape[0]}"
        )
