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


# def posterior(Xnew, X, y, likelihood_noise, gp):
#     L, alpha = get_factorisations(X, y, likelihood_noise, gp)
#     K_Xx = gp.kernel(Xnew, X)
#     # Calculate the Mean
#     mu_y = jnp.dot(K_Xx, alpha)
#     # =====================================
#     # 5. PREDICTIVE COVARIANCE DISTRIBUTION
#     # =====================================
#     v = cho_solve(L, K_Xx.T)
#     # Calculate kernel matrix for inputs
#     K_xx = gp.kernel(Xnew, Xnew)
#     cov_y = K_xx - jnp.dot(K_Xx, v)
#     return mu_y, cov_y
