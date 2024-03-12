from jax import config

config.update("jax_enable_x64", True)

import jax.random as jr

from gpjax import kernels


class Kernels:
    param_names = ["n_data", "dimensionality"]
    params = [[10, 100, 500, 1000, 2000], [1, 2, 5]]

    def setup(self, n_datapoints: int, n_dims: int):
        key = jr.key(123)
        self.X = jr.uniform(
            key=key, minval=-3.0, maxval=3.0, shape=(n_datapoints, n_dims)
        )


class RBF(Kernels):
    def setup(self, n_datapoints: int, n_dims: int):
        super().setup(n_datapoints, n_dims)
        self.kernel = kernels.RBF(active_dims=list(range(n_dims)))

    def time_covfunc_call(self, n_datapoints: int, n_dims: int):
        self.kernel.gram(self.X)


class Matern12(Kernels):
    def setup(self, n_datapoints: int, n_dims: int):
        super().setup(n_datapoints, n_dims)
        self.kernel = kernels.Matern12(active_dims=list(range(n_dims)))

    def time_covfunc_call(self, n_datapoints: int, n_dims: int):
        self.kernel.gram(self.X)


class Matern32(Kernels):
    def setup(self, n_datapoints: int, n_dims: int):
        super().setup(n_datapoints, n_dims)
        self.kernel = kernels.Matern32(active_dims=list(range(n_dims)))

    def time_covfunc_call(self, n_datapoints: int, n_dims: int):
        self.kernel.gram(self.X)


class Matern52(Kernels):
    def setup(self, n_datapoints: int, n_dims: int):
        super().setup(n_datapoints, n_dims)
        self.kernel = kernels.Matern52(active_dims=list(range(n_dims)))

    def time_covfunc_call(self, n_datapoints: int, n_dims: int):
        self.kernel.gram(self.X)


class PoweredExponential(Kernels):
    def setup(self, n_datapoints: int, n_dims: int):
        super().setup(n_datapoints, n_dims)
        self.kernel = kernels.PoweredExponential(active_dims=list(range(n_dims)))

    def time_covfunc_call(self, n_datapoints: int, n_dims: int):
        self.kernel.gram(self.X)


class RationalQuadratic(Kernels):
    def setup(self, n_datapoints: int, n_dims: int):
        super().setup(n_datapoints, n_dims)
        self.kernel = kernels.RationalQuadratic(active_dims=list(range(n_dims)))

    def time_covfunc_call(self, n_datapoints: int, n_dims: int):
        self.kernel.gram(self.X)


class Polynomial(Kernels):
    def setup(self, n_datapoints: int, n_dims: int):
        super().setup(n_datapoints, n_dims)
        self.kernel = kernels.Polynomial(active_dims=list(range(n_dims)))

    def time_covfunc_call(self, n_datapoints: int, n_dims: int):
        self.kernel.gram(self.X)


class Linear(Kernels):
    def setup(self, n_datapoints: int, n_dims: int):
        super().setup(n_datapoints, n_dims)
        self.kernel = kernels.Linear(active_dims=list(range(n_dims)))

    def time_covfunc_call(self, n_datapoints: int, n_dims: int):
        self.kernel.gram(self.X)


class ArcCosine(Kernels):
    def setup(self, n_datapoints: int, n_dims: int):
        super().setup(n_datapoints, n_dims)
        self.kernel = kernels.ArcCosine(active_dims=list(range(n_dims)))

    def time_covfunc_call(self, n_datapoints: int, n_dims: int):
        self.kernel.gram(self.X)
