import jax.numpy as jnp
from objax import Module
from typing import Callable
from jax import vmap, nn
from .parameters import Parameter


class Kernel(Module):
    def __init__(self,
                 name: str = None):
        # TODO: Make the transformation a method of the parameter type.
        self.name = name
        self.spectral = False

    @staticmethod
    def gram(func: Callable, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        mapx1 = vmap(lambda x, y: func(x=x, y=y),
                     in_axes=(0, None),
                     out_axes=0)
        mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
        return mapx2(x, y)

    @staticmethod
    def dist(x: jnp.array, y: jnp.array) -> float:
        return jnp.sum((x - y)**2)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray):
        raise NotImplementedError


class Stationary(Kernel):
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "Stationary"):
        super().__init__(name=name)
        self.lengthscale = Parameter(lengthscale)
        self.variance = Parameter(variance)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class RBF(Stationary):
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "RBF"):
        super().__init__(lengthscale=lengthscale,
                         variance=variance,
                         name=name)

    def feature_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        ell = self.lengthscale.transformed
        sigma = self.variance.transformed
        tau = self.dist(x, y)
        return sigma * jnp.exp(-tau/ell)

    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return self.gram(self.feature_map, X, Y).squeeze()
