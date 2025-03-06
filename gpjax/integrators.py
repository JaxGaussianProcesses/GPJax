import abc

import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import Float
import numpy as np

from gpjax.typing import Array

L = tp.TypeVar(
    "L",
    bound="gpjax.likelihoods.AbstractLikelihood",  # noqa: F821
)
GL = tp.TypeVar("GL", bound="gpjax.likelihoods.Gaussian")  # noqa: F821
HGL = tp.TypeVar("HGL", bound="gpjax.likelihoods.HeteroscedasticGaussian")  # noqa: F821


class AbstractIntegrator:
    r"""Base class for integrators."""

    @abc.abstractmethod
    def integrate(
        self,
        fun: tp.Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        likelihood: L | None,
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
        fun: tp.Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        likelihood: L | None,
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


class GHQuadratureIntegrator(AbstractIntegrator):
    r"""Compute an integral using Gauss-Hermite quadrature.

    Gauss-Hermite quadrature is a method for approximating integrals through a
    weighted sum of function evaluations at specific points
    $$
    \int F(t)\exp(-t^2)\mathrm{d}t \approx \sum_{j=1}^J w_j F(t_j)
    $$
    where $t_j$ and $w_j$ are the roots and weights of the $J$-th order Hermite
    polynomial $H_J(t)$ that we can look up in table
    [link](https://keisan.casio.com/exec/system/1281195844).
    """

    def __init__(self, num_points: int = 20):
        r"""Initialize the integrator.

        Args:
            num_points (int, optional): The number of points to use in the
                quadrature. Defaults to 20.
        """
        self.num_points = num_points

    def integrate(
        self,
        fun: tp.Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        likelihood: L | None,
    ) -> Float[Array, " N"]:
        r"""Compute a quadrature integral.

        Args:
            fun: the likelihood to be integrated.
            y: the observed response variable.
            mean: the mean of the variational distribution.
            variance: the variance of the variational distribution.
            likelihood: the likelihood function.

        Returns:
            The expected log likelihood as an array of shape (N,).
        """
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.num_points)
        sd = jnp.sqrt(variance)
        X = mean + jnp.sqrt(2.0) * sd * gh_points
        W = gh_weights / jnp.sqrt(jnp.pi)
        val = jnp.sum(fun(X, y) * W, axis=1)
        return val


class AnalyticalGaussianIntegrator(AbstractIntegrator):
    r"""Compute the analytical integral of a Gaussian likelihood.

    When the likelihood function is Gaussian, the integral can be computed in closed
    form. For a Gaussian likelihood $p(y|f) = \mathcal{N}(y|f, \sigma^2)$ and a
    variational distribution $q(f) = \mathcal{N}(f|m, s)$, the expected
    log-likelihood is given by
    $$
    \mathbb{E}_{q(f)}[\log p(y|f)] = -\frac{1}{2}\left(\log(2\pi\sigma^2) + \frac{1}{\sigma^2}((y-m)^2 + s)\right)
    $$
    """

    def integrate(
        self,
        fun: tp.Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        likelihood: GL,
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
        obs_stddev = likelihood.obs_stddev.value.squeeze()
        sq_error = jnp.square(y - mean)
        log2pi = jnp.log(2.0 * jnp.pi)
        val = jnp.sum(
            log2pi + jnp.log(obs_stddev**2) + (sq_error + variance) / obs_stddev**2,
            axis=1,
        )
        return -0.5 * val


class HeteroscedasticGaussianIntegrator(AbstractIntegrator):
    r"""Compute the expected log likelihood for a heteroscedastic Gaussian likelihood.

    This integrator approximates the expected log likelihood for a heteroscedastic Gaussian
    likelihood where the noise variance is parameterized by a latent GP.

    For a heteroscedastic Gaussian likelihood with signal f and log noise variance g:
    p(y|f,g) = N(y|f, exp(g))

    And variational distributions:
    q(f) = N(f|m_f, s_f)
    q(g) = N(g|m_g, s_g)

    We approximate the expected log likelihood using a first-order Taylor expansion
    of the exp(g) term around the mean of g.
    """

    def integrate(
        self,
        fun: tp.Callable,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        likelihood: HGL,
    ) -> Float[Array, " N"]:
        r"""Compute the expected log likelihood for heteroscedastic Gaussian.

        Args:
            fun (Callable): The heteroscedastic Gaussian likelihood to be integrated.
            y (Float[Array, 'N D']): The observed response variable.
            mean (Float[Array, 'N D']): The variational mean for both signal and noise.
            variance (Float[Array, 'N D']): The variational variance for both signal and noise.
            likelihood (HeteroscedasticGaussian): The heteroscedastic Gaussian likelihood.

        Returns:
            Float[Array, 'N']: The expected log likelihood.
        """
        # Split the inputs for signal and noise
        n = mean.shape[0] // 2

        # Signal mean and variance
        mean_f = mean[:n]
        var_f = variance[:n]

        # Log noise variance mean and variance
        mean_g = mean[n:]
        var_g = variance[n:]

        # Clip log noise variance to avoid numerical issues
        mean_g_clipped = jnp.clip(mean_g, likelihood.clip_min, likelihood.clip_max)

        # Compute expected noise variance: E[exp(g)] ≈ exp(E[g]) * (1 + 0.5*Var[g])
        # This is a second-order Taylor approximation of E[exp(g)]
        expected_noise_var = jnp.exp(mean_g_clipped) * (1.0 + 0.5 * var_g)

        # Compute expected log likelihood
        log2pi = jnp.log(2.0 * jnp.pi)
        sq_error = jnp.square(y - mean_f)

        # E[log p(y|f,g)] ≈ -0.5 * (log(2π) + E[g] + (y-E[f])²/E[exp(g)] + Var[f]/E[exp(g)])
        val = jnp.sum(
            log2pi + mean_g_clipped + (sq_error + var_f) / expected_noise_var,
            axis=1,
        )

        return -0.5 * val


__all__ = [
    "AbstractIntegrator",
    "GHQuadratureIntegrator",
    "AnalyticalGaussianIntegrator",
    "HeteroscedasticGaussianIntegrator",
]
