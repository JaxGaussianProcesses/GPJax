from jax import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import jax.random as jr

import gpjax as gpx
from gpjax.fit import get_batch


class Sparse:
    param_names = ["n_data", "n_inducing", "batch_size"]
    params = [[10000, 20000, 50000], [10, 20, 50, 100, 200], [32, 64, 128, 256]]

    def setup(self, n_datapoints: int, n_inducing: int, batch_size: int):
        key = jr.key(123)
        self.X = jr.normal(key=key, shape=(n_datapoints, 1))
        self.y = jnp.sin(self.X[:, :1])
        self.data = gpx.Dataset(X=self.X, y=self.y)
        kernel = gpx.kernels.RBF(active_dims=list(range(1)))
        meanf = gpx.mean_functions.Constant()
        self.prior = gpx.gps.Prior(kernel=kernel, mean_function=meanf)
        self.likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.data.n)
        self.posterior = self.prior * self.likelihood

        Z = jnp.linspace(self.X.min(), self.X.max(), n_inducing).reshape(-1, 1)
        self.q = gpx.variational_families.VariationalGaussian(
            posterior=self.posterior, inducing_inputs=Z
        )
        self.objective = gpx.objectives.ELBO(negative=True)

    def time_eval(self, n_datapoints: int, n_dims: int, batch_size: int):
        key = jr.key(123)
        batch = get_batch(train_data=self.data, batch_size=batch_size, key=key)
        self.objective(self.q, batch)

    def time_grad(self, n_datapoints: int, n_dims: int, batch_size: int):
        key = jr.key(123)
        batch = get_batch(train_data=self.data, batch_size=batch_size, key=key)
        jax.grad(self.objective)(self.q, batch)
