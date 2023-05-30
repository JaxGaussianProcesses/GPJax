from abc import abstractmethod
from dataclasses import dataclass

from beartype.typing import (
    Any,
    Callable,
)
import jax.numpy as jnp
from jaxtyping import Float
import numpy as np
from simple_pytree import Pytree

from gpjax.typing import Array


@dataclass
class AbstractIntegrator(Pytree):
    @abstractmethod
    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        sigma2: Float[Array, "N D"],
        **likelihood_params: Any,
    ):
        raise NotImplementedError("self.integrate not implemented")

    def __call__(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        sigma2: Float[Array, "N D"],
        *args: Any,
        **kwargs: Any,
    ):
        return self.integrate(fun, y, mean, sigma2, *args, **kwargs)


@dataclass
class GHQuadratureIntegrator(AbstractIntegrator):
    num_points: int = 20

    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        sigma2: Float[Array, "N D"],
        **likelihood_params: Any,
    ) -> Float[Array, " N"]:
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.num_points)
        sd = jnp.sqrt(sigma2)
        X = mean + jnp.sqrt(2.0) * sd * gh_points
        W = gh_weights / jnp.sqrt(jnp.pi)
        val = jnp.sum(fun(X, y) * W, axis=1)
        return val


@dataclass
class AnalyticalGaussianIntegrator(AbstractIntegrator):
    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        sigma2: Float[Array, "N D"],
        **likelihood_params: Any,
    ) -> Float[Array, " N"]:
        obs_noise = likelihood_params["obs_noise"].squeeze()
        sq_error = jnp.square(y - mean)
        log2pi = jnp.log(2.0 * jnp.pi)
        val = jnp.sum(
            log2pi + jnp.log(obs_noise) + (sq_error + sigma2) / obs_noise, axis=1
        )
        return -0.5 * val


__all__ = [
    "AbstractIntegrator",
    "GHQuadratureIntegrator",
    "AnalyticalGaussianIntegrator",
]
