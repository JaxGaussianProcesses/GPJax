import jax.numpy as jnp
import jax.random as jr
from mean_functions import mean
from multipledispatch import dispatch
from .types import Prior, Posterior
from .kernel import jitter_matrix, gram


@dispatch(jnp.array, Prior, jnp.array)
def sample(key: jnp.array, gp: Prior, sample_points: jnp.array, n_samples: int = 1):
    mu = mean(gp.mean_func, sample_points)
    jitter = jitter_matrix(sample_points.shape[0], 1e-6)
    Sigma = gram(gp.kernel, sample_points) + jitter
    return jr.multivariate_normal(key, mu, Sigma, shape=(n_samples, ))


@dispatch(Posterior, jnp.array, int)
def sample(gp: Prior, X: jnp.array, n_samples:int, jitter_amount:float=1e-6):
    print("Posterior sample")