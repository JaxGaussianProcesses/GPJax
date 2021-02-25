from chex import dataclass
from typing import Optional
from .types import Array
import jax.numpy as jnp
from jax import vmap
from multipledispatch import dispatch


@dataclass
class Kernel:
    stationary: Optional[bool] = False
    spectral: Optional[bool] = False
    name: Optional[str] = 'Kernel'

    def __call__(self, x: Array, y: Array) -> Array:
        raise NotImplementedError


@dataclass
class RBF(Kernel):
    stationary = True
    spectral = False
    name = 'RBF'

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(-0.5 * squared_distance(x, y))


@dispatch(RBF)
def initialise(kernel: RBF):
    return {'lengthscale': jnp.array([1.0]),
            'variance': jnp.array([1.0])
            }


def squared_distance(x: Array, y: Array):
    return jnp.sum((x - y)**2)


def gram(kernel: Kernel, inputs: Array) -> Array:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1))(inputs))(inputs)


def cross_covariance(kernel: Kernel, x: Array, y: Array) -> Array:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1))(x))(y)
