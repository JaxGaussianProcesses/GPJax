from .likelihoods import Gaussian, Likelihood
from .kernel import Kernel
from .mean_functions import MeanFunction
from .mean_functions import ZeroMean
from .utils import get_factorisations
from objax import TrainVar, Module
import jax.numpy as jnp
from jax import nn
from jax.scipy.linalg import cho_solve
import jax.random as jr
from jax.scipy.stats import multivariate_normal


class Prior(Module):
    def __init__(self,
                 kernel: Kernel,
                 mean_function: MeanFunction = ZeroMean(),
                 observation_noise: jnp.ndarray = jnp.array([0.1]),
                 jitter: float = 1e-6):
        self.meanf = mean_function
        self.kernel = kernel
        self.noise = TrainVar(observation_noise)
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
        return Posterior(self, other)


class Posterior(Module):
    def __init__(self, prior: Prior, likelihood: Gaussian):
        self.kernel = prior.kernel
        self.meanf = prior.meanf
        self.likelihood = likelihood
        self.jitter = prior.jitter

    def forward(self, X: jnp.ndarray, y) -> jnp.ndarray:
        Inn = jnp.eye(X.shape[0])
        mu = self.meanf(X)
        cov = self.kernel(X, X) + self.jitter * Inn
        cov += nn.softplus(self.likelihood.noise.value) * Inn
        # L = jnp.linalg.cholesky(cov)
        # TODO: Return the logpdf w.r.t. the Cholesky, not the full cov.
        lpdf = multivariate_normal.logpdf(y.squeeze(), mu.squeeze(), cov)
        return lpdf

    def predict(self, Xstar, X, y):
        sigma = nn.softplus(self.likelihood.noise.value)
        L, alpha = get_factorisations(X, y, sigma, self.kernel, self.meanf)
        Kfx = self.kernel(Xstar, X)
        mu = jnp.dot(Kfx, alpha)
        v = cho_solve(L, Kfx.T)
        # Calculate kernel matrix for inputs
        Kxx = self.kernel(Xstar, Xstar)
        cov = Kxx - jnp.dot(Kfx, v)
        return mu, cov
