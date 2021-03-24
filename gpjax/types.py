from typing import Optional, Union

import jax.numpy as jnp
from chex import dataclass
from jax.interpreters.ad import JVPTracer
from jax.interpreters.partial_eval import DynamicJaxprTracer
from jax.interpreters.pxla import ShardedDeviceArray

NoneType = type(None)
Array = (jnp.ndarray, ShardedDeviceArray, jnp.DeviceArray, JVPTracer, DynamicJaxprTracer)
Arrays = Union[jnp.ndarray, ShardedDeviceArray, jnp.DeviceArray, JVPTracer, DynamicJaxprTracer]


@dataclass(repr=False)
class Dataset:
    X: Arrays
    y: Arrays = None

    def __post_init__(self):
        assert (
            self.X.ndim == 2
        ), f"2-dimensional training inputs are required. Current dimension: {self.X.ndim}."
        if self.y != None:
            assert (
                self.y.ndim == 2
            ), f"2-dimensional training outputs are required. Current dimension: {self.y.ndim}."
            assert (
                self.X.shape[0] == self.y.shape[0]
            ), f"Number of inputs must equal the number of outputs. \nCurrent counts:\n- X: {self.X.shape[0]}\n- y: {self.y.shape[0]}"

    def __repr__(self) -> str:
        return f"- Number of datapoints: {self.X.shape[0]}\n- Dimension: {self.X.shape[1]}"

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        return self.y.shape[1]
