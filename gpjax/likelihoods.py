from objax import Module
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from .parameters import Parameter
from typing import Optional
from tensorflow_probability.substrates.jax import distributions as tfd


class Likelihood(Module):
    """
    Base class for all likelihood functions. By inheriting the `Module` class from Objax, seamless interaction with
    model parameters is provided.
    """
    def __init__(self, name: Optional[str] = "Likelihood"):
        """
        Args:
            name: Optional naming of the likelihood.
        """
        self.name = name

    def __call__(self):
        raise NotImplementedError


class Gaussian(Likelihood):
    """
    Gaussian likelihood function 
    """
    def __init__(self, noise: jnp.array = jnp.array([1.0]), name="Gaussian"):
        super().__init__(name=name)
        self.noise = Parameter(noise)

    def log_likelihood(self, x: jnp.ndarray, mu: jnp.ndarray, L: jnp.ndarray):
        delta = x - mu
        alpha = solve_triangular(L, delta, lower=True)
        L_diag = jnp.sum(jnp.log(jnp.diag(L)))
        ll = -0.5 * jnp.sum(jnp.square(alpha), axis=0)
        ll -= 0.5 * jnp.log(2 * jnp.pi)
        ll -= L_diag
        return ll

    def __call__(self):
        raise NotImplementedError


class Bernoulli(Likelihood):
    def __init__(self):
        super().__init__(name="Bernoulli")
        self.random_variable = tfd.ProbitBernoulli

    def log_density(self, F, Y):
        return self.random_variable(F).log_prob(Y)

    def mean(self, probs):
        return self.random_variable()
