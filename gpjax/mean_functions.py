import jax.numpy as jnp
from objax import Module
from objax.typing import JaxArray


class MeanFunction(Module):
    def __init__(self, name: str = "Mean Function"):
        self.name = name

    def __call__(self, X: JaxArray):
        raise NotImplementedError


class ZeroMean(MeanFunction):
    def __init__(self, name: str = "Zero Mean"):
        super().__init__(name=name)

    def __call__(self, X: JaxArray) -> JaxArray:
        return jnp.zeros((X.shape[0], 1), dtype=X.dtype)
