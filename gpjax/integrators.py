from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from beartype.typing import Callable
import jax.numpy as jnp
from jaxtyping import Float
import numpy as np

from gpjax.typing import Array

if TYPE_CHECKING:
    import gpjax.likelihoods


@dataclass
class AbstractIntegrator:
    r"""Base class for integrators."""

    @abstractmethod
    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        likelihood: "gpjax.likelihoods.AbstractLikelihood" = None,
    ) -> Float[Array, " N"]:
        r"""Integrate a function with respect to a Gaussian distribution.

        Typically, the function will be the likelihood function and the mean
        and variance will be the parameters of the variational distribution.

        Args:
            fun (Callable): The function to be integrated.
            y (Float[Array, 'N D']): The observed response variable.
            mean (Float[Array, 'N D']): The mean of the variational distribution.
            variance (Float[Array, 'N D']): The variance of the variational
                distribution.
            likelihood (AbstractLikelihood): The likelihood function.
        """
        raise NotImplementedError("self.integrate not implemented")

    def __call__(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        likelihood: "gpjax.likelihoods.AbstractLikelihood" = None,
    ) -> Float[Array, " N"]:
        r"""Integrate a function with respect to a Gaussian distribution.

        Typically, the function will be the likelihood function and the mean
        and variance will be the parameters of the variational distribution.

        Args:
            fun (Callable): The function to be integrated.
            y (Float[Array, 'N D']): The observed response variable.
            mean (Float[Array, 'N D']): The mean of the variational distribution.
            variance (Float[Array, 'N D']): The variance of the variational
                distribution.
            likelihood (AbstractLikelihood): The likelihood function.
        """
        return self.integrate(fun, y, mean, variance, likelihood)


@dataclass
class GHQuadratureIntegrator(AbstractIntegrator):
    r"""Compute an integral using Gauss-Hermite quadrature.

    Gauss-Hermite quadrature is a method for approximating integrals through a
    weighted sum of function evaluations at specific points
    ```math
    \int F(t)\exp(-t^2)\mathrm{d}t \approx \sum_{j=1}^J w_j F(t_j)
    ```
    where $`t_j`$ and $`w_j`$ are the roots and weights of the $`J`$-th order Hermite
    polynomial $`H_J(t)`$ that we can look up in table
    [link](https://keisan.casio.com/exec/system/1281195844).
    """
    num_points: int = 20

    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        likelihood: "gpjax.likelihoods.AbstractLikelihood" = None,
    ) -> Float[Array, " N"]:
        r"""Compute a quadrature integral.

        Args:
            fun (Callable): The likelihood to be integrated.
            y (Float[Array, 'N D']): The observed response variable.
            mean (Float[Array, 'N D']): The mean of the variational distribution.
            variance (Float[Array, 'N D']): The variance of the variational
                distribution.
            likelihood (AbstractLikelihood): The likelihood function.

        Returns:
            Float[Array, 'N']: The expected log likelihood.
        """
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.num_points)
        sd = jnp.sqrt(variance)
        X = mean + jnp.sqrt(2.0) * sd * gh_points
        W = gh_weights / jnp.sqrt(jnp.pi)
        val = jnp.sum(fun(X, y) * W, axis=1)
        return val


@dataclass
class AnalyticalGaussianIntegrator(AbstractIntegrator):
    r"""Compute the analytical integral of a Gaussian likelihood.

    When the likelihood function is Gaussian, the integral can be computed in closed
    form. For a Gaussian likelihood $`p(y|f) = \mathcal{N}(y|f, \sigma^2)`$ and a
    variational distribution $`q(f) = \mathcal{N}(f|m, s)`$, the expected
    log-likelihood is given by
    ```math
    \mathbb{E}_{q(f)}[\log p(y|f)] = -\frac{1}{2}\left(\log(2\pi\sigma^2) + \frac{1}{\sigma^2}\mathbb{E}_{q(f)}[(y-f)^2 + s]\right)
    """

    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        likelihood: "gpjax.likelihoods.Gaussian" = None,
    ) -> Float[Array, " N"]:
        r"""Compute a Gaussian integral.

        Args:
            fun (Callable): The Gaussian likelihood to be integrated.
            y (Float[Array, 'N D']): The observed response variable.
            mean (Float[Array, 'N D']): The mean of the variational distribution.
            variance (Float[Array, 'N D']): The variance of the variational
                distribution.
            likelihood (Gaussian): The Gaussian likelihood function.

        Returns:
            Float[Array, 'N']: The expected log likelihood.
        """
        obs_noise = likelihood.obs_noise.squeeze()
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
