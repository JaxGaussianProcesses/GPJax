import jax.numpy as jnp
from jax.nn import softplus
from .utilities import JaxArray


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
    def forward(x: JaxArray) -> JaxArray:
        return jnp.log(jnp.exp(x) - 1.)

    @staticmethod
    def backward(x: jnp.ndarray) -> jnp.ndarray:
        return softplus(x)


class Identity(Transform):
    def __init__(self):
        super().__init__(name='Identity')

    @staticmethod
    def forward(x: JaxArray) -> JaxArray:
        return x

    @staticmethod
    def backward(x: JaxArray) -> JaxArray:
        return x
