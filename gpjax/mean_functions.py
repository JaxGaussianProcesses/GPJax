from typing import Optional

import jax.numpy as jnp
from chex import dataclass
from multipledispatch import dispatch

from .types import Array


@dataclass
class MeanFunction:
    output_dim: Optional[int] = 1
    name: Optional[str] = "Mean Function"

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError


@dataclass
class Zero(MeanFunction):
    output_dim: Optional[int] = 1
    name: Optional[str] = "Zero mean"

    def __call__(self, x: Array) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.zeros(shape=out_shape)


@dispatch(Zero)
def initialise(meanf: Zero) -> dict:
    return {}
