from jax.config import config

config.update("jax_enable_x64", True)

from gpjax import kernels
import jax.random as jr

KERNEL_NAMES = ["RBF", "MATERN12", "MATERN32", "MATERN52", "POLYNOMIAL"]

N_DATAPOINTS = [10, 100, 1000]
N_DIMS = [1, 2, 5]


def get_kernel(kernel_name: str, n_dims: int):
    active_dims = list(range(n_dims))
    if kernel_name == "RBF":
        return kernels.RBF(active_dims=active_dims)
    elif kernel_name == "MATERN12":
        return kernels.Matern12(active_dims=active_dims)
    elif kernel_name == "MATERN32":
        return kernels.Matern32(active_dims=active_dims)
    elif kernel_name == "MATERN52":
        return kernels.Matern52(active_dims=active_dims)
    elif kernel_name == "POLYNOMIAL":
        return kernels.Polynomial(active_dims=active_dims)
    else:
        raise ValueError("Unknown covariance function name")


class Kernels:
    param_names = ["kernel", "n_data", "dimensionality"]
    params = [KERNEL_NAMES, N_DATAPOINTS, N_DIMS]

    def setup(self, kernel_func, n_datapoints, n_dims):
        key = jr.PRNGKey(123)
        self.X = jr.uniform(
            key=key, minval=-3.0, maxval=3.0, shape=(n_datapoints, n_dims)
        )
        self.kernelfunc = get_kernel(kernel_name=kernel_func, n_dims=n_dims)

    def time_covfunc_call(self, kernel_func, n_datapoints, n_dims):
        self.kernelfunc.gram(self.X)
