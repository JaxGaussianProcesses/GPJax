from jax.config import config

config.update("jax_enable_x64", True)
import gpjax as gpx
from gpjax.gps import ConjugatePosterior
import jax.numpy as jnp
import jax.random as jr
import jax

N_TEST = [100, 200, 500, 1000, 2000, 3000]
N_DIMS = [1, 2, 5]
LIKELIHOOD = ["Gaussian", "Bernoulli"]


class Objectives:
    param_names = [
        "n_test",
        "n_dims",
        "likelihood",
    ]
    params = [N_TEST, N_DIMS, LIKELIHOOD]

    def setup(self, n_test, n_dims, likelihood):
        key = jr.PRNGKey(123)
        self.X = jr.normal(key=key, shape=(100, n_dims))
        self.y = jnp.sin(self.X[:, :1])
        if likelihood == "Bernoulli":
            self.y = jnp.where(self.y > 0, 1, 0)
        self.data = gpx.Dataset(X=self.X, y=self.y)
        kernel = gpx.kernels.RBF(active_dims=list(range(n_dims)))
        meanf = gpx.mean_functions.Constant()
        self.prior = gpx.Prior(kernel=kernel, mean_function=meanf)
        if likelihood == "Bernoulli":
            self.likelihood = gpx.likelihoods.Bernoulli(num_datapoints=self.data.n)
        elif likelihood == "Gaussian":
            self.likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.data.n)
        self.posterior: ConjugatePosterior = self.prior * self.likelihood
        key, subkey = jr.split(key)
        self.xtest = jr.normal(key=subkey, shape=(n_test, n_dims))

    def time_predict(self, n_test, n_dims, likelihood):
        self.posterior.predict(test_inputs=self.xtest, train_data=self.data)
