import abc
from math import degrees
from typing import Optional

import jax.numpy as jnp
from chex import dataclass
from jax import vmap

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
        return (
            f"{self.name}:\n\t Stationary: {self.stationary}\n\t Spectral form:"
            f" {self.spectral} \n\t ARD structure: {self.ard}"
        )

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

    def __call__(
        self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict
    ) -> Array:
        x /= params["lengthscale"]
        y /= params["lengthscale"]
        K = params["variance"] * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()

    @property
    def params(self) -> dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


@dataclass(repr=False)
class Matern12(Kernel):
    ndims: Optional[int] = 1
    stationary: Optional[bool] = True
    spectral: Optional[bool] = False
    name: Optional[str] = "Matern 1/2"

    def __call__(
        self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict
    ) -> Array:
        x /= params["lengthscale"]
        y /= params["lengthscale"]
        K = params["variance"] * jnp.exp(-0.5 * euclidean_distance(x, y))
        return K.squeeze()

    @property
    def params(self) -> dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


@dataclass(repr=False)
class Matern32(Kernel):
    ndims: Optional[int] = 1
    stationary: Optional[bool] = True
    spectral: Optional[bool] = False
    name: Optional[str] = "Matern 3/2"

    def __call__(
        self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict
    ) -> Array:
        x /= params["lengthscale"]
        y /= params["lengthscale"]
        tau = euclidean_distance(x, y)
        K = (
            params["variance"]
            * (1.0 + jnp.sqrt(3.0) * tau)
            * jnp.exp(-jnp.sqrt(3.0) * tau)
        )
        return K.squeeze()

    @property
    def params(self) -> dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


@dataclass(repr=False)
class Matern52(Kernel):
    ndims: Optional[int] = 1
    stationary: Optional[bool] = True
    spectral: Optional[bool] = False
    name: Optional[str] = "Matern 5/2"

    def __call__(
        self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict
    ) -> Array:
        x /= params["lengthscale"]
        y /= params["lengthscale"]
        tau = euclidean_distance(x, y)
        K = (
            params["variance"]
            * (1.0 + jnp.sqrt(5.0) * tau + 5.0 / 3.0 * jnp.square(tau))
            * jnp.exp(-jnp.sqrt(5.0) * tau)
        )
        return K.squeeze()

    @property
    def params(self) -> dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


@dataclass(repr=False)
class Polynomial(Kernel):
    ndims: Optional[int] = 1
    stationary: Optional[bool] = False
    spectral: Optional[bool] = False
    name: Optional[str] = "Polynomial"
    degree: int = 1

    def __post_init__(self):
        self.name = f"Polynomial Degree: {self.degree}"

    def __call__(
        self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict
    ) -> Array:
        K = jnp.power(
            params["shift"] + jnp.dot(x * params["variance"], y),
            self.degree,
        )
        return K.squeeze()

    @property
    def params(self) -> dict:
        return {
            "shift": jnp.array([1.0]),
            "variance": jnp.array([1.0] * self.ndims),
        }


def squared_distance(x: Array, y: Array):
    return jnp.sum((x - y) ** 2)


def euclidean_distance(x: Array, y: Array):
    return jnp.sum(jnp.abs(x - y))


def gram(kernel: Kernel, inputs: Array, params: dict) -> Array:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(inputs))(
        inputs
    )


def cross_covariance(kernel: Kernel, x: Array, y: Array, params: dict) -> Array:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, params))(x))(y)
