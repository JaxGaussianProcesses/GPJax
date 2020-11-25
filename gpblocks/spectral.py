from typing import Callable
import jax.numpy as jnp
import jax.random as jr
from .kernel import RBF
from jax import nn
from objax import TrainVar


class SpectralRBF(RBF):
    def __init__(self,
                 num_basis: int,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 parameter_transform: Callable = nn.softplus,
                 key = jr.PRNGKey(123),
                 name: str = "RBF"):
        super().__init__(parameter_transform=parameter_transform,
                         lengthscale=lengthscale,
                         variance=variance,
                         name=name)
        self.num_basis = num_basis
        self.features = TrainVar(jr.normal(key, shape=(self.num_basis,)))
