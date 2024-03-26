from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    TypeVar,
    Union,
)

from beartype.typing import Callable
import jax.numpy as jnp
from jaxtyping import Float
import numpy as np

import gpjax
from gpjax.typing import Array

Likelihood = TypeVar(
    "Likelihood",
    bound=Union["gpjax.likelihoods.AbstractLikelihood", None],  # noqa: F821
)
Gaussian = TypeVar("Gaussian", bound="gpjax.likelihoods.Gaussian")  # noqa: F821


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
        likelihood: Likelihood,
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
        likelihood: Likelihood,
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
    num_points: int = 100

    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N 1"],
        mean: Float[Array, "L N"],
        variance: Float[Array, "L N"],
        likelihood: Likelihood,
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
        assert jnp.shape(mean)[0]==1, "This integrator only works for single latents"
        assert jnp.shape(variance)[0]==1, "This integrator only works for single latents"
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.num_points) # [n] [n]
        sd = jnp.sqrt(variance) # [L N]
        X = mean[:,:,None] + jnp.sqrt(2.0) * sd[:,:,None] * gh_points[None, None, :] # [L N n]
        X = jnp.transpose(X, (1,0,2)) # [N 1 n]
        W = gh_weights / jnp.sqrt(jnp.pi) # [n]
        val = jnp.sum(fun(X, y) * W[None,:], axis=1)
        return val




@dataclass
class TwoDimGHQuadratureIntegrator(GHQuadratureIntegrator):
    num_points_per_dim: int = 10
    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N 1"],
        mean: Float[Array, "L N"],
        variance: Float[Array, "L N"],
        likelihood: Likelihood,
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
        assert jnp.shape(mean)[0]==2, "This integrator only works for 2d latents"
        assert jnp.shape(variance)[0]==2, "This integrator only works for 2d latents"
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.num_points_per_dim) # [n] [n]
        sd = jnp.sqrt(variance) # [L N]
        X = mean[:,:,None] + jnp.sqrt(2.0) * sd[:,:,None] * gh_points[None, None, :] # [L N n]
        X = jnp.transpose(X, (1,0,2)) # [N L n]
        X = X[:,:,:,None] + X[:,:,None,:] #[N L n n]
        X = jnp.reshape(X, (jnp.shape(X)[0], jnp.shape(X)[1], -1))  # [N L n*n]
        W = gh_weights / (jnp.sqrt(jnp.pi)**2) #[n]
        W = jnp.repeat(W[None,:],self.num_points_per_dim,0) * jnp.repeat(W[:, None],self.num_points_per_dim,1) #[n n]
        W = jnp.reshape(W, (1,-1)) # [1, n*n]
        val = jnp.sum(fun(X, y)* W, axis=1)
        return val


@dataclass
class ThreeDimGHQuadratureIntegrator(GHQuadratureIntegrator):
    num_points_per_dim: int = 25
    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N 1"],
        mean: Float[Array, "L N"],
        variance: Float[Array, "L N"],
        likelihood: Likelihood,
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
        assert jnp.shape(mean)[0]==3, "This integrator only works for 2d latents"
        assert jnp.shape(variance)[0]==3, "This integrator only works for 2d latents"
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.num_points_per_dim) # [n] [n]
        sd = jnp.sqrt(variance) # [L N]
        X = mean[:,:,None] + jnp.sqrt(2.0) * sd[:,:,None] * gh_points[None, None, :] # [L N n]
        X = jnp.transpose(X, (1,0,2)) # [N L n]
        X = X[:,:,:,None,None] + X[:,:,None,None,:] + X[:,:,None,:,None] #[N L n n n]
        X = jnp.reshape(X, (jnp.shape(X)[0], jnp.shape(X)[1], -1))  # [N L n*n*n]
        W = gh_weights / (jnp.sqrt(jnp.pi)**3) #[n]
    
        W = (
            jnp.repeat(jnp.repeat(W[None,:],self.num_points_per_dim,0)[None,:,:], self.num_points_per_dim,0)
            * jnp.repeat(jnp.repeat(W[:,None],self.num_points_per_dim,1)[None,:,:], self.num_points_per_dim,0)
            * jnp.repeat(jnp.repeat(W[:, None],self.num_points_per_dim,1)[:,:, None], self.num_points_per_dim,-1)
        )#[n n n]
        W = jnp.reshape(W, (1,-1)) # [1, n*n*n]
        val = jnp.sum(fun(X, y)* W, axis=1)
        return val



@dataclass
class AnalyticalGaussianIntegrator(AbstractIntegrator):
    r"""Compute the analytical integral of a Gaussian likelihood.

    When the likelihood function is Gaussian, the integral can be computed in closed
    form. For a Gaussian likelihood $`p(y|f) = \mathcal{N}(y|f, \sigma^2)`$ and a
    variational distribution $`q(f) = \mathcal{N}(f|m, s)`$, the expected
    log-likelihood is given by
    ```math
    \mathbb{E}_{q(f)}[\log p(y|f)] = -\frac{1}{2}\left(\log(2\pi\sigma^2) + \frac{1}{\sigma^2}((y-m)^2 + s)\right)
    ```
    """

    def integrate(
        self,
        fun: Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        likelihood: Gaussian,
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
        obs_stddev = likelihood.obs_stddev.squeeze()
        sq_error = jnp.square(y - mean)
        log2pi = jnp.log(2.0 * jnp.pi)
        val = jnp.sum(
            log2pi + jnp.log(obs_stddev**2) + (sq_error + variance) / obs_stddev**2,
            axis=1,
        )
        return -0.5 * val


__all__ = [
    "AbstractIntegrator",
    "GHQuadratureIntegrator",
    "AnalyticalGaussianIntegrator",
]
