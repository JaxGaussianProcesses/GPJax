from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import gpjax as gpx


class Gaussian:
    param_names = [
        "n_test",
        "n_dims",
    ]
    params = [[100, 200, 500, 1000, 2000, 3000], [1, 2, 5]]

    def setup(self, n_test: int, n_dims: int):
        key = jr.key(123)
        self.X = jr.normal(key=key, shape=(100, n_dims))
        self.y = jnp.sin(self.X[:, :1])
        self.data = gpx.Dataset(X=self.X, y=self.y)
        kernel = gpx.kernels.RBF(active_dims=list(range(n_dims)))
        meanf = gpx.mean_functions.Constant()
        self.prior = gpx.gps.Prior(kernel=kernel, mean_function=meanf)
        self.likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.data.n)
        self.posterior = self.prior * self.likelihood
        key, subkey = jr.split(key)
        self.xtest = jr.normal(key=subkey, shape=(n_test, n_dims))

    def time_predict(self, n_test: int, n_dims: int):
        self.posterior.predict(test_inputs=self.xtest, train_data=self.data)


class Bernoulli:
    param_names = [
        "n_test",
        "n_dims",
    ]
    params = [[100, 200, 500, 1000, 2000, 3000], [1, 2, 5]]

    def setup(self, n_test: int, n_dims: int):
        key = jr.key(123)
        self.X = jr.normal(key=key, shape=(100, n_dims))
        self.y = jnp.sin(self.X[:, :1])
        self.y = jnp.array(jnp.where(self.y > 0, 1, 0), dtype=jnp.float64)
        self.data = gpx.Dataset(X=self.X, y=self.y)
        kernel = gpx.kernels.RBF(active_dims=list(range(n_dims)))
        meanf = gpx.mean_functions.Constant()
        self.prior = gpx.gps.Prior(kernel=kernel, mean_function=meanf)
        self.likelihood = gpx.likelihoods.Bernoulli(num_datapoints=self.data.n)
        self.posterior = self.prior * self.likelihood
        key, subkey = jr.split(key)
        self.xtest = jr.normal(key=subkey, shape=(n_test, n_dims))

    def time_predict(self, n_test: int, n_dims: int):
        self.posterior.predict(test_inputs=self.xtest, train_data=self.data)


class Poisson:
    param_names = [
        "n_test",
        "n_dims",
    ]
    params = [[100, 200, 500, 1000, 2000, 3000], [1, 2, 5]]

    def setup(self, n_test: int, n_dims: int):
        key = jr.key(123)
        self.X = jr.normal(key=key, shape=(100, n_dims))
        f = lambda x: 2.0 * jnp.sin(3 * x) + 0.5 * x  # latent function
        self.y = jnp.array(jr.poisson(key, jnp.exp(f(self.X))), dtype=jnp.float64)
        self.data = gpx.Dataset(X=self.X, y=self.y)
        kernel = gpx.kernels.RBF(active_dims=list(range(n_dims)))
        meanf = gpx.mean_functions.Constant()
        self.prior = gpx.gps.Prior(kernel=kernel, mean_function=meanf)
        self.likelihood = gpx.likelihoods.Bernoulli(num_datapoints=self.data.n)
        self.posterior = self.prior * self.likelihood
        key, subkey = jr.split(key)
        self.xtest = jr.normal(key=subkey, shape=(n_test, n_dims))

    def time_predict(self, n_test: int, n_dims: int):
        self.posterior.predict(test_inputs=self.xtest, train_data=self.data)
