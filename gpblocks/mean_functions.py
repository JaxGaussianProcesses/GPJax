from objax import Module, TrainVar
import jax.numpy as jnp


class MeanFunction(Module):
    def __init__(self, name: str = "Mean Function"):
        self.name = name


class ZeroMean(MeanFunction):
    def __init__(self, name: str = "Zero Mean"):
        super().__init__(name=name)

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros(X.shape, dtype=X.dtype)
