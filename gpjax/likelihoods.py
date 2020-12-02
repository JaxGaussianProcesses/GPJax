from objax import Module, TrainVar
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


class Likelihood(Module):
    def __init__(self, name: str = "Likelihood"):
        self.name = name


class Gaussian(Likelihood):
    def __init__(self, noise: jnp.array = jnp.array([1.0]), name="Gaussian"):
        super().__init__(name=name)
        self.noise = TrainVar(noise)

    def log_likelihood(self, x: jnp.ndarray, mu: jnp.ndarray, L: jnp.ndarray):
        delta = x - mu
        n_dimension = delta.shape[0]
        alpha = solve_triangular(L, delta, lower=True)
        L_diag = jnp.sum(jnp.log(jnp.diag(L)))
        ll = -0.5 * jnp.sum(jnp.square(alpha), axis=0)
        ll -= 0.5 * jnp.log(2 * jnp.pi)
        ll -= L_diag
        return ll
