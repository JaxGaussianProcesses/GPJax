from typing import Optional
import jax.numpy as jnp
from jax.interpreters.pxla import ShardedDeviceArray
from jax.interpreters.ad import JVPTracer
from jax.interpreters.partial_eval import DynamicJaxprTracer
from chex import dataclass

NoneType = type(None)
Array = (jnp.ndarray, ShardedDeviceArray, jnp.DeviceArray, JVPTracer, DynamicJaxprTracer)


@dataclass(repr=False)
class Dataset:
    X: Array
    y: Array
    # assert (
    #         y.ndim == 2
    # ), f"2-dimensional training outputs are required. Current dimensional: {y.ndim}."

    def __repr__(self) -> str:
        return f"- Number of datapoints: {self.X.shape[0]}\n- Dimension: {self.X.shape[1]}"

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def __len__(self) -> int:
        return self.X.shape[0]
