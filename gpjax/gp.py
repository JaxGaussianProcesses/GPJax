from .likelihoods import Gaussian, Likelihood
from .kernel import Kernel
from .mean_functions import MeanFunction
from .mean_functions import ZeroMean
from .utils import get_factorisations
from objax import TrainVar, Module
import jax.numpy as jnp
from jax import nn
from jax.scipy.linalg import cho_solve, solve_triangular
import jax.random as jr
from jax.scipy.stats import multivariate_normal


class Prior(Module):
    def __init__(self,
                 kernel: Kernel,
                 mean_function: MeanFunction = ZeroMean(),
                 jitter: float = 1e-6):
        self.meanf = mean_function
        self.kernel = kernel
        self.jitter = jitter

    def sample(self, X: jnp.ndarray, key, n_samples: int = 1):
        Inn = jnp.eye(X.shape[0])
        mu = self.meanf(X)
        cov = self.kernel(X, X) + self.jitter * Inn
        return jr.multivariate_normal(key,
                                      mu.squeeze(),
                                      cov,
                                      shape=(n_samples, ))

    def __mul__(self, other: Likelihood):
        if self.kernel.spectral is True:
            return SpectralPosterior(self, other)
        else:
            return Posterior(self, other)


class Posterior(Module):
    def __init__(self, prior: Prior, likelihood: Gaussian):
        self.kernel = prior.kernel
        self.meanf = prior.meanf
        self.likelihood = likelihood
        self.jitter = prior.jitter

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        Inn = jnp.eye(X.shape[0])
        mu = self.meanf(X)
        cov = self.kernel(X, X) + self.jitter * Inn
        cov += nn.softplus(self.likelihood.noise.value) * Inn
        # L = jnp.linalg.cholesky(cov)
        # TODO: Return the logpdf w.r.t. the Cholesky, not the full cov.
        lpdf = multivariate_normal.logpdf(y.squeeze(), mu.squeeze(), cov)
        return lpdf

    def neg_mll(self, X: jnp.ndarray, y: jnp.ndarray):
        return -self.marginal_ll(X, y)

    def predict(self, Xstar, X, y):
        sigma = nn.softplus(self.likelihood.noise.value)
        L, alpha = get_factorisations(X, y, sigma, self.kernel, self.meanf)
        Kfx = self.kernel(Xstar, X)
        mu = jnp.dot(Kfx, alpha)
        v = cho_solve(L, Kfx.T)
        Kxx = self.kernel(Xstar, Xstar)
        cov = Kxx - jnp.dot(Kfx, v)
        return mu, cov


class SpectralPosterior(Posterior):
    def __init__(self, prior: Prior, likelihood: Gaussian):
        super().__init__(prior, likelihood)

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        N = X.shape[0]
        m = self.kernel.num_basis
        l_var = self.likelihood.noise.value
        k_var = self.kernel.variance.value
        phi = self.kernel(X, self.kernel.features.T)
        A = (k_var / m) * jnp.matmul(phi, phi.T) + l_var * jnp.eye(m * 2)
        Rt = jnp.linalg.cholesky(A)
        RtiPhit = solve_triangular(Rt, phi.T)
        RtiPhity = jnp.matmul(RtiPhit, y)

        term1 = (jnp.sum(y**2) -
                 jnp.sum(RtiPhity**2) * k_var / m) * 0.5 / l_var
        term2 = jnp.sum(jnp.log(jnp.diag(
            Rt.T))) + (N * 0.5 - m) * jnp.log(l_var) + (N * 0.5 *
                                                        jnp.log(2 * jnp.pi))
        return term1 + term2
