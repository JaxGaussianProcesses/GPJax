from __future__ import annotations
import typing

import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor, solve_triangular
from objax import Module
from tensorflow_probability.substrates.jax import distributions as tfd
from typing import Tuple

from ..likelihoods import Gaussian, Likelihood
from ..parameters import Parameter
from ..transforms import Identity

if typing.TYPE_CHECKING:
    from .priors import Prior


class Posterior(Module):
    """
    A base class for GP posterior
    """
    def __init__(self, prior: Prior, likelihood: Likelihood):
        self.kernel = prior.kernel
        self.meanf = prior.meanf
        self.likelihood = likelihood
        self.jitter = prior.jitter

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def neg_mll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def predict(self, Xstar, X, y):
        raise NotImplementedError

    def __eq__(self, other: Posterior):
        return list(self.vars().items()) == list(other.vars().items())


class PosteriorExact(Posterior):
    """
    A Gaussian process posterior whereby the likelihood function is a Gaussian distribution.
    """
    def __init__(self, prior: Prior, likelihood: Gaussian):
        super().__init__(prior, likelihood)

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Here we compute :math:`\log p(y | x, \theta)
        Args:
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A multivariate normal distribution

        """
        Inn = jnp.eye(X.shape[0])
        mu = self.meanf(X)
        cov = self.kernel(X, X) + self.jitter * Inn
        cov += self.likelihood.noise.untransform * Inn
        L = jnp.linalg.cholesky(cov)
        return tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)

    def neg_mll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        In gradient based optimisation of the marginal log-likelihood (MLL), we minimise the negative MLL.
        Args:
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A unary float array

        """
        rv = self.marginal_ll(X, y)
        log_prior_density = jnp.sum(jnp.array([v.log_density for k, v in self.vars().items()]))
        return -(rv.log_prob(y.squeeze()).mean()+log_prior_density)

    def predict(self, Xstar, X, y):
        """
        Conditional upon the GP posterior, compute the predictive posterior given a set of new and unseen test points.
        Args:
            Xstar: The new inputs that correspond to the locations that we'd like to sample the GP predictive posterior at
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A predictive mean and predictive variance

        """
        # Compute covariance matrices and jitter matrix
        Kff = self.kernel(X, X)
        Kfx = self.kernel(Xstar, X)
        Kxx = self.kernel(Xstar, Xstar)
        Inn = jnp.eye(X.shape[0])
        # Compute prior mean
        mu_f = self.meanf(X)
        # Realise the current estimate of the observational noise
        sigma = self.likelihood.noise.untransform
        # Compute the lower Cholesky decomposition
        L = cho_factor(Kff + Inn * sigma, lower=True)
        err = y.reshape(-1, 1) - mu_f.reshape(-1, 1)
        weights = cho_solve(L, err)
        # Compute the predictive mean
        mu = jnp.dot(Kfx, weights)
        # Compute the predictive variance
        v = cho_solve(L, Kfx.T)
        cov = Kxx - jnp.dot(Kfx, v)
        return mu, cov


class PosteriorApprox(Posterior):
    def __init__(self, prior: Prior, likelihood: Likelihood):
        super().__init__(prior=prior, likelihood=likelihood)
        self.n = None
        self.nu = None
        self.latent_init = False

    def _initialise_latent_values(self, n):
        self.n = n
        self.nu = Parameter(jnp.zeros(shape=(self.n, 1)), transform=Identity(), prior=tfd.Normal(loc=0., scale=1.))
        self.latent_init = True

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        if not self.latent_init:
            self._initialise_latent_values(X.shape[0])
        mu = self.meanf(X)
        Kxx = self.kernel(X, X)
        Inn = jnp.eye(self.n) * self.jitter
        L = jnp.linalg.cholesky(Kxx + Inn)
        F = jnp.matmul(L, self.nu.untransform) + mu
        ll = jnp.sum(self.likelihood.log_density(F, y))
        lpd = 0
        for k, v in self.vars().items():
            lpd += jnp.sum(v.log_density).reshape()
        # lpd = jnp.sum(jnp.array([v.log_density for k, v in self.vars().items()]))
        return ll + lpd

    def neg_mll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return -self.marginal_ll(X, y)

    def predict(self, Xstar: jnp.ndarray, X:jnp.ndarray, y:jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if not self.latent_init:
            self._initialise_latent_values(X.shape[0])
        Inn = jnp.eye(X.shape[0])*self.jitter
        Kff = self.kernel(X, X) + Inn
        L = jnp.linalg.cholesky(Kff)
        Kfx = self.kernel(X, Xstar)
        Kxx = self.kernel(Xstar, Xstar)
        A = solve_triangular(L, Kfx)
        cov = Kxx - jnp.matmul(A.T, A)
        mean = jnp.matmul(A.T, self.nu.untransform)
        return mean, cov