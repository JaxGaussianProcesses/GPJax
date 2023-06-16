from jax.config import config

config.update("jax_enable_x64", True)

from gpjax.linops import DenseLinearOperator
import jax.numpy as jnp
from sklearn.datasets import make_spd_matrix
import jax.random as jr


class LinOps:
    param_names = ["n_data"]
    params = [[10, 100, 200, 500, 1000]]

    def setup(self, n_datapoints):
        key = jr.PRNGKey(123)
        self.X = jnp.asarray(make_spd_matrix(n_dim=n_datapoints, random_state=123))
        self.y = jr.normal(key=key, shape=(n_datapoints, 1))
        self.linop = DenseLinearOperator(matrix=self.X)

    def time_root(self, n_datapoints):
        self.linop.to_root()

    def time_inverse(self, n_datapoints):
        self.linop.inverse()

    def time_logdet(self, n_datapoints):
        self.linop.log_det()

    def time_solve(self, n_datapoints):
        self.linop.solve(self.y)
