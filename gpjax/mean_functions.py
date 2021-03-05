from typing import Optional

import jax.numpy as jnp
from chex import dataclass
from multipledispatch import dispatch

from .types import Array


@dataclass(repr=False)
class MeanFunction:
    output_dim: Optional[int] = 1
    name: Optional[str] = "Mean function"

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}\n\t Output dimension: {self.output_dim}"


@dataclass(repr=False)
class Zero(MeanFunction):
    output_dim: Optional[int] = 1
    name: Optional[str] = "Zero mean function"

    def __call__(self, x: Array) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.zeros(shape=out_shape)


@dispatch(Zero)
def initialise(meanf: Zero) -> dict:
    return {}
