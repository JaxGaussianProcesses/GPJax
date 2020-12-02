import jax.numpy as jnp
from objax import TrainVar, Module
from typing import Callable
from jax import vmap, nn


class Kernel(Module):
    def __init__(self,
                 parameter_transform: Callable = nn.softplus,
                 name: str = None):
        # TODO: Make the transformation a method of the parameter type.
        self.transform = parameter_transform
        self.name = name

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
                 parameter_transform: Callable = nn.softplus,
                 name: str = "Stationary"):
        super().__init__(parameter_transform=parameter_transform, name=name)
        self.lengthscale = TrainVar(lengthscale)
        self.variance = TrainVar(variance)


class RBF(Stationary):
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 parameter_transform: Callable = nn.softplus,
                 name: str = "RBF"):
        super().__init__(parameter_transform=parameter_transform,
                         lengthscale=lengthscale,
                         variance=variance,
                         name=name)

    def feature_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        ell = self.transform(self.lengthscale.value)
        sigma = self.transform(self.variance.value)
        tau = self.dist(x, y)
        return sigma * jnp.exp(-tau / ell)

    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return self.gram(self.feature_map, X, Y).squeeze()
