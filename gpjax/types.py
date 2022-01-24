import typing as tp

import jax.numpy as jnp
import numpy as np
from chex import dataclass

NoneType = type(None)
Array = tp.Union[np.ndarray, jnp.ndarray]


@dataclass(repr=False)
class Dataset:
    X: Array
    y: Array = None

    def __repr__(self) -> str:
        return (
            f"- Number of datapoints: {self.X.shape[0]}\n- Dimension:"
            f" {self.X.shape[1]}"
        )

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        return self.y.shape[1]


def verify_dataset(ds: Dataset) -> None:
    assert ds.X.ndim == 2, (
        "2-dimensional training inputs are required. Current dimension:"
        f" {ds.X.ndim}."
    )
    if ds.y is not None:
        assert ds.y.ndim == 2, (
            "2-dimensional training outputs are required. Current dimension:"
            f" {ds.y.ndim}."
        )
        assert ds.X.shape[0] == ds.y.shape[0], (
            "Number of inputs must equal the number of outputs. \nCurrent"
            f" counts:\n- X: {ds.X.shape[0]}\n- y: {ds.y.shape[0]}"
        )
