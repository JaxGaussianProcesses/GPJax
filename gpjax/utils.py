import jax.numpy as jnp
from typing import Tuple
from jax.scipy.linalg import cho_factor, cho_solve
from .mean_functions import MeanFunction
# from .kernel import Kernel


def cholesky_factorisation(K: jnp.ndarray,
                           Y: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
    L = cho_factor(K, lower=True)
    weights = cho_solve(L, Y)
    return L, weights


def get_factorisations(
        X: jnp.ndarray, Y: jnp.ndarray, likelihood_noise: float,
        kernel,
        meanf: MeanFunction) -> Tuple[Tuple[jnp.ndarray, bool], jnp.ndarray]:
    Inn = jnp.eye(X.shape[0])
    mu_x = meanf(X)
    Kxx = kernel(X, X)
    return cholesky_factorisation(
        Kxx + likelihood_noise * Inn,
        Y.reshape(-1, 1) - mu_x.reshape(-1, 1),
    )

#
# def softplus(x):
#     return jnp.log(jnp.exp(x) - 1.)
#
#
# class Transform:
#     def __init__(self, name="Transform"):
#         self.name = name
#
#     @staticmethod
#     def forward(x):
#         raise NotImplementedError
#
#     @staticmethod
#     def reverse(x):
#         raise NotImplementedError
#
#
# class LogTransform(Transform):
#     def __init__(self, name='Log-transform'):
#         super().__init__(name=name)
#
#     @staticmethod
#     def forward(x):
#         return jnp.log(x)
#
#     @staticmethod
#     def reverse(x):
#         return jnp.exp(x)
#
#
# class Identity(Transform):
#     def __init__(self, name='Identity-transform'):
#         super().__init__(name=name)
#
#     @staticmethod
#     def forward(x):
#         return x
#
#     @staticmethod
#     def reverse(x):
#         return x
