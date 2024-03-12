from jax import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import jax.random as jr

import gpjax as gpx


class Gaussian:
    param_names = [
        "n_data",
        "n_dims",
    ]
    params = [[10, 100, 200, 500, 1000], [1, 2, 5]]

    def setup(self, n_datapoints: int, n_dims: int):
        key = jr.key(123)
        self.X = jr.normal(key=key, shape=(n_datapoints, n_dims))
        self.y = jnp.sin(self.X[:, :1])
        self.data = gpx.Dataset(X=self.X, y=self.y)
        kernel = gpx.kernels.RBF(active_dims=list(range(n_dims)))
        meanf = gpx.mean_functions.Constant()
        self.prior = gpx.gps.Prior(kernel=kernel, mean_function=meanf)
        self.likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.data.n)
        self.objective = gpx.objectives.ConjugateMLL()
        self.posterior = self.prior * self.likelihood

    def time_eval(self, n_datapoints: int, n_dims: int):
        self.objective.step(self.posterior, self.data).block_until_ready()

    def time_grad(self, n_datapoints: int, n_dims: int):
        jax.block_until_ready(jax.grad(self.objective.step)(self.posterior, self.data))


class Bernoulli:
    param_names = [
        "n_data",
        "n_dims",
    ]
    params = [[10, 100, 200, 500, 1000], [1, 2, 5]]

    def setup(self, n_datapoints: int, n_dims: int):
        key = jr.key(123)
        self.X = jr.normal(key=key, shape=(n_datapoints, n_dims))
        self.y = jnp.where(jnp.sin(self.X[:, :1]) > 0, 1, 0)
        self.data = gpx.Dataset(X=self.X, y=self.y)
        kernel = gpx.kernels.RBF(active_dims=list(range(n_dims)))
        meanf = gpx.mean_functions.Constant()
        self.prior = gpx.gps.Prior(kernel=kernel, mean_function=meanf)
        self.likelihood = gpx.likelihoods.Bernoulli(num_datapoints=self.data.n)
        self.objective = gpx.objectives.LogPosteriorDensity()
        self.posterior = self.prior * self.likelihood

    def time_eval(self, n_datapoints: int, n_dims: int):
        self.objective.step(self.posterior, self.data).block_until_ready()

    def time_grad(self, n_datapoints: int, n_dims: int):
        jax.block_until_ready(jax.grad(self.objective.step)(self.posterior, self.data))


class Poisson:
    param_names = [
        "n_data",
        "n_dims",
    ]
    params = [[10, 100, 200, 500, 1000], [1, 2, 5]]

    def setup(self, n_datapoints: int, n_dims: int):
        key = jr.key(123)
        self.X = jr.normal(key=key, shape=(n_datapoints, n_dims))
        f = lambda x: 2.0 * jnp.sin(3 * x) + 0.5 * x  # latent function
        self.y = jr.poisson(key, jnp.exp(f(self.X)))
        self.data = gpx.Dataset(X=self.X, y=self.y)
        kernel = gpx.kernels.RBF(active_dims=list(range(n_dims)))
        meanf = gpx.mean_functions.Constant()
        self.prior = gpx.gps.Prior(kernel=kernel, mean_function=meanf)
        self.likelihood = gpx.likelihoods.Poisson(num_datapoints=self.data.n)
        self.objective = gpx.objectives.LogPosteriorDensity()
        self.posterior = self.prior * self.likelihood

    def time_eval(self, n_datapoints: int, n_dims: int):
        self.objective.step(self.posterior, self.data).block_until_ready()

    def time_grad(self, n_datapoints: int, n_dims: int):
        jax.block_until_ready(jax.grad(self.objective.step)(self.posterior, self.data))
