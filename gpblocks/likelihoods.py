from objax import Module, TrainVar
import jax.numpy as jnp


class Likelihood(Module):
    def __init__(self, name: str = "Likelihood"):
        self.name = name


class Gaussian(Likelihood):
    def __init__(self, noise: jnp.array = jnp.array([1.0]), name="Gaussian"):
        super().__init__(name=name)
        self.noise = TrainVar(noise)

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros(X.shape[-1], dtype=X.dtype)
