import jax.numpy as jnp
from typing import Tuple
from jax.scipy.linalg import cho_factor, cho_solve
from .mean_functions import MeanFunction
from .kernel import Kernel


def cholesky_factorisation(K: jnp.ndarray,
                           Y: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
    L = cho_factor(K, lower=True)
    weights = cho_solve(L, Y)
    return L, weights


def get_factorisations(
        X: jnp.ndarray, Y: jnp.ndarray, likelihood_noise: float,
        kernel: Kernel,
        meanf: MeanFunction) -> Tuple[Tuple[jnp.ndarray, bool], jnp.ndarray]:
    Inn = jnp.eye(X.shape[0])
    mu_x = meanf(X)
    Kxx = kernel(X, X)
    return cholesky_factorisation(
        Kxx + likelihood_noise * Inn,
        Y.reshape(-1, 1) - mu_x.reshape(-1, 1),
    )
