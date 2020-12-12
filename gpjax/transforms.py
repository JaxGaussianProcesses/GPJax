import jax.numpy as jnp
from jax.nn import softplus


class Transform:
    def __init__(self, name="Transformation"):
        self.name = name

    @staticmethod
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class Softplus(Transform):
    def __init__(self):
        super().__init__(name='Softplus')

    @staticmethod
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        return softplus(x)

    @staticmethod
    def backward(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(jnp.exp(x)-1.)
