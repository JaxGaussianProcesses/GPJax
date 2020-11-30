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
                 key=jr.PRNGKey(123),
                 name: str = "RBF"):
        super().__init__(parameter_transform=parameter_transform,
                         lengthscale=lengthscale,
                         variance=variance,
                         name=name)
        self.input_dim = lengthscale.shape[
            0]  # TODO: This assumes the lengthscale is ARD. This value should be driven by the data's dimension instead.
        self.num_basis = num_basis
        self.features = TrainVar(
            jr.normal(key, shape=(self.num_basis, self.input_dim)))

    def _compute_phi(self, X: jnp.ndarray):
        """
        Takes an NxD matrix and returns a 2*NxM matrix

        :param X:
        :return:
        """
        cos_freqs = jnp.cos(X.dot(self.features.T))
        sin_freqs = jnp.sin(X.dot(self.features.T))
        phi = jnp.vstack((cos_freqs, sin_freqs))
        return phi

    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        phi_matrix = self._compute_phi(X)
        return phi_matrix.dot(phi_matrix.T)
