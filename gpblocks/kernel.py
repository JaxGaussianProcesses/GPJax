import jax.numpy as jnp
from objax import TrainVar, Module
from typing import Callable
from jax import vmap, nn


class Kernel(Module):
    def __init__(self, name: str = None):
        self.name = name

    @staticmethod
    def covariance_matrix(func: Callable, x: jnp.ndarray,
                          y: jnp.ndarray) -> jnp.ndarray:
        mapx1 = vmap(lambda x, y: func(x=x, y=y),
                     in_axes=(0, None),
                     out_axes=0)
        mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
        return mapx2(x, y)

    @staticmethod
    def sqeuclidean_distance(x: jnp.array, y: jnp.array) -> float:
        return jnp.sum((x - y)**2)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray):
        raise NotImplementedError


class Stationary(Kernel):
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([0.1]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "Stationary"):
        super().__init__(name=name)
        self.lengthscale = TrainVar(lengthscale)
        self.variance = TrainVar(variance)


class RBF(Stationary):
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([0.1]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "Stationary"):
        super().__init__(lengthscale=lengthscale, variance=variance, name=name)

    def feature_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        ell = nn.softplus(self.lengthscale.value)
        sigma = nn.softplus(self.variance.value)
        x = x / ell
        y = y / ell

        # return the ard kernel
        tau = self.sqeuclidean_distance(x, y)
        return sigma * jnp.exp(-tau)

    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return self.covariance_matrix(self.feature_map, X, Y).squeeze()


#
# def rbf_kernel(gamma: float, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
#     return jnp.exp(-gamma * sqeuclidean_distance(x, y))
#
#
# class RBFKernel(Module):
#     def __init__(self):
#         self.gamma = TrainVar(jnp.array([0.1]))
#
#     def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
#         kernel_func = partial(rbf_kernel, gamma=self.gamma.value)
#         return covariance_matrix(kernel_func, X, Y).squeeze()
#
#
# class ARDKernel(Module):
#     def __init__(self):
#         self.length_scale = TrainVar(jnp.array([0.1]))
#         self.amplitude = TrainVar(jnp.array([1.]))
#
#     def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
#         kernel_func = partial(ard_kernel,
#                               length_scale=nn.softplus(
#                                   self.length_scale.value),
#                               amplitude=nn.softplus(self.amplitude.value))
#         return covariance_matrix(kernel_func, X, Y).squeeze()
#
#
# def covariance_matrix(
#     func: Callable,
#     x: jnp.ndarray,
#     y: jnp.ndarray,
# ) -> jnp.ndarray:
#     mapx1 = vmap(lambda x, y: func(x=x, y=y), in_axes=(0, None), out_axes=0)
#     mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
#     return mapx2(x, y)
#
#
# def ard_kernel(x: jnp.ndarray, y: jnp.ndarray, length_scale,
#                amplitude) -> jnp.ndarray:
#     x = x / length_scale
#     y = y / length_scale
#
#     # return the ard kernel
#     return amplitude * jnp.exp(-sqeuclidean_distance(x, y))
#
#
# def sqeuclidean_distance(x: jnp.array, y: jnp.array) -> float:
#     return jnp.sum((x - y)**2)
