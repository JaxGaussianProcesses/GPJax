from typing import Optional

import jax.numpy as jnp
from chex import dataclass
from jax import vmap
import abc

from gpjax.types import Array


@dataclass(repr=False)
class Kernel:
    ndims: Optional[int] = 1
    stationary: Optional[bool] = False
    spectral: Optional[bool] = False
    name: Optional[str] = "Kernel"

    @abc.abstractmethod
    def __call__(self, x: Array, y: Array, params: dict) -> Array:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}:\n\t Stationary: {self.stationary}\n\t Spectral form: {self.spectral} \n\t ARD structure: {self.ard}"

    @property
    def ard(self):
        return True if self.ndims > 1 else False

    @property
    @abc.abstractmethod
    def params(self) -> dict:
        raise NotImplementedError


@dataclass(repr=False)
class RBF(Kernel):
    ndims: Optional[int] = 1
    stationary: Optional[bool] = True
    spectral: Optional[bool] = False
    name: Optional[str] = "Radial basis function kernel"

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        x /= params["lengthscale"]
        y /= params["lengthscale"]
        K = params["variance"] * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()

    @property
    def params(self) -> dict:
        return {"lengthscale": jnp.array([1.0] * self.ndims), "variance": jnp.array([1.0])}


def squared_distance(x: Array, y: Array):
    return jnp.sum((x - y) ** 2)


def gram(kernel: Kernel, inputs: Array, params: dict) -> Array:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(inputs))(inputs)


def cross_covariance(kernel: Kernel, x: Array, y: Array, params: dict) -> Array:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(x))(y)
