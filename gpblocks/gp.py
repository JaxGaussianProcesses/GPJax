import jax.numpy as jnp
from .types import Posterior
from .kernel import jitter_matrix, gram


def marginal_log_likelihood(model: Posterior, X, y):
    """
    Compute the marginal log-likelihood of a GP with Gaussian likelihood

    :param model: A GP posterior struct
    :param X: Some observerved 2-dimensional inputs
    :param y: Observed 2-dimensional outputs
    :return: A floating point value of the current marginal log-likelihood
    """
    N = X.shape[0]
    jitter = jitter_matrix(N, model.likelihood.obs_noise)
    Kxx = gram(model.prior.kernel, X) + jitter
    L = jnp.linalg.cholesky(Kxx)
    diag_sum = jnp.sum(jnp.diag(L))
    alpha = jnp.linalg.solve(jnp.transpose(L), jnp.linalg.solve(L, y))
    return -0.5 * jnp.squeeze(jnp.matmul(jnp.transpose(y), alpha)) - diag_sum - 0.5 * N * jnp.log(2 * jnp.pi)


