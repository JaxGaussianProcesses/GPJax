import abc
from typing import Dict, Optional

import jax.numpy as jnp
from chex import dataclass

from .types import Array


@dataclass(repr=False)
class MeanFunction:
    output_dim: Optional[int] = 1
    name: Optional[str] = "Mean function"

    @abc.abstractmethod
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}\n\t Output dimension: {self.output_dim}"

    @property
    @abc.abstractmethod
    def params(self) -> dict:
        raise NotImplementedError


@dataclass(repr=False)
class Zero(MeanFunction):
    output_dim: Optional[int] = 1
    name: Optional[str] = "Zero mean function"

    def __call__(self, x: Array, params: dict) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.zeros(shape=out_shape)

    @property
    def params(self) -> dict:
        return {}


@dataclass(repr=False)
class Constant(MeanFunction):
    output_dim: Optional[int] = 1
    name: Optional[str] = "Constant mean function"

    def __call__(self, x: Array, params: Dict) -> Array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.ones(shape=out_shape) * params["variance"]

    @property
    def params(self) -> dict:
        return {"variance": jnp.array(1.0)}
