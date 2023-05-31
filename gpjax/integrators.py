from abc import abstractmethod
from dataclasses import dataclass

from beartype.typing import (
    Any,
    Callable,
)
import jax.numpy as jnp
from jaxtyping import Float
import numpy as np

from gpjax.typing import Array


@dataclass
class AbstractIntegrator:
    """Base class for integrators."""

    @abstractmethod
    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        **likelihood_params: Any,
    ):
        """Integrate a function with respect to a Gaussian distribution.

        Typically, the function will be the likelihood function and the mean
        and variance will be the parameters of the variational distribution.

        Args:
            fun (Callable): The function to be integrated.
            y (Float[Array, 'N D']): _description_
            mean (Float[Array, 'N D']): _description_
            variance (Float[Array, 'N D']): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("self.integrate not implemented")

    def __call__(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        **likelihood_params: Any,
    ):
        return self.integrate(fun, y, mean, variance, **likelihood_params)


@dataclass
class GHQuadratureIntegrator(AbstractIntegrator):
    num_points: int = 20

    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        **likelihood_params: Any,
    ) -> Float[Array, " N"]:
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.num_points)
        sd = jnp.sqrt(variance)
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
        variance: Float[Array, "N D"],
        **likelihood_params: Any,
    ) -> Float[Array, " N"]:
        obs_noise = likelihood_params["obs_noise"].squeeze()
        sq_error = jnp.square(y - mean)
        log2pi = jnp.log(2.0 * jnp.pi)
        val = jnp.sum(
            log2pi + jnp.log(obs_noise) + (sq_error + variance) / obs_noise, axis=1
        )
        return -0.5 * val


__all__ = [
    "AbstractIntegrator",
    "GHQuadratureIntegrator",
    "AnalyticalGaussianIntegrator",
]
