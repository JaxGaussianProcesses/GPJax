import tensorflow as tf
import numpy as np
from multipledispatch import dispatch
from .gp import Prior, Posterior
from .types import InputData


@dispatch(Prior, np.ndarray, int)
def sample(gp: Prior, X: np.ndarray, n_samples:int, jitter_amount:float=1e-6):
    mu = gp.mean_func(X)
    k = gp.kernel(X, X, jitter_amount=jitter_amount)
    # TODO: Possible idea return a tfp distribution instead of a set of samples
    samples = np.random.multivariate_normal(tf.squeeze(mu), k, size=n_samples)
    return samples


@dispatch(Posterior, np.ndarray, int)
def sample(gp: Prior, X: np.ndarray, n_samples:int, jitter_amount:float=1e-6):
    print("Posterior sample")