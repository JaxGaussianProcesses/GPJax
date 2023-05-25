from abc import abstractmethod
from simple_pytree import Pytree
from dataclasses import dataclass
from beartype.typing import Callable, Any, Union
import jax.numpy as jnp
from jaxtyping import Float
import numpy as np
from gpjax.typing import Array, ScalarFloat


@dataclass
class AbstractIntegrator(Pytree):
    def __call__(self, *args: Any, **kwargs: Any):
        return self.integrate(*args, **kwargs)

    @abstractmethod
    def integrate(self, fun: Callable, *args: Any, **kwargs: Any):
        raise NotImplementedError("self.integrate not implemented")


@dataclass
class GHQuadratureIntegrator(AbstractIntegrator):
    num_points: int = 20

    def integrate(
        self,
        fun: Callable,
        mean: Float[Array, "N D"],
        sd: Float[Array, "N D"],
        *args,
        **kwargs,
    ) -> Float[Array, " N"]:
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.num_points)
        X = mean + jnp.sqrt(2.0) * sd * gh_points
        W = gh_weights / jnp.sqrt(jnp.pi)
        return jnp.sum(fun(X, kwargs["y"]) * W, axis=1)


@dataclass
class AnalyticalGaussianIntegrator(AbstractIntegrator):
    def integrate(
        self,
        fun: Callable,
        mean: Float[Array, "N D"],
        sd: Float[Array, "N D"],
        *args,
        **kwargs,
    ) -> Float[Array, " N"]:
        variance = jnp.square(sd)
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * jnp.log(kwargs["obs_noise"])
            - 0.5 * ((kwargs["y"] - mean) ** 2 + variance) / variance,
            axis=-1,
        )


__all__ = [
    "AbstractIntegrator",
    "GHQuadratureIntegrator",
    "AnalyticalGaussianIntegrator",
]
