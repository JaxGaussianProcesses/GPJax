from jax.config import config

config.update("jax_enable_x64", True)
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import jax

N_DATAPOINTS = [10, 100, 200, 500, 1000]
N_DIMS = [1, 2, 5]
OBS = ["Gaussian", "Bernoulli"]


class Objectives:
    param_names = ["n_data", "n_dims", "likelihood"]
    params = [N_DATAPOINTS, N_DIMS, OBS]

    def setup(self, n_datapoints, n_dims, obs):
        key = jr.PRNGKey(123)
        self.X = jr.normal(key=key, shape=(n_datapoints, n_dims))
        self.y = jnp.sin(self.X[:, :1])
        self.data = gpx.Dataset(X=self.X, y=self.y)
        kernel = gpx.kernels.RBF(active_dims=list(range(n_dims)))
        meanf = gpx.mean_functions.Constant()
        self.prior = gpx.Prior(kernel=kernel, mean_function=meanf)
        self.likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.data.n)
        self.posterior = self.prior * self.likelihood
        self.objective = gpx.ConjugateMLL()

    def time_eval(self, n_datapoints, n_dims, obs):
        self.objective(self.posterior, train_data=self.data)

    def time_grad(self, n_datapoints, n_dims, obs):
        jax.grad(self.objective)(self.posterior, train_data=self.data)
