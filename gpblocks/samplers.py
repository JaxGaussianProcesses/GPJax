import tensorflow as tf
import numpy as np
from multipledispatch import dispatch
from .types import Prior, Posterior


@dispatch(Prior, np.ndarray, int)
def sample(gp: Prior, X: np.ndarray, n_samples:int, jitter_amount:float=1e-6):
    """
    Draw samples from the GP prior i.e.

    :param gp:
    :param X:
    :param n_samples:
    :param jitter_amount:
    :return:
    """
    mu = gp.mean_func(X)
    k = gp.kernel(X, X, jitter_amount=jitter_amount)
    # TODO: Possible idea return a tfp distribution instead of a set of samples
    samples = np.random.multivariate_normal(tf.squeeze(mu), k, size=n_samples)
    return samples


@dispatch(Posterior, np.ndarray, int)
def sample(gp: Prior, X: np.ndarray, n_samples:int, jitter_amount:float=1e-6):
    print("Posterior sample")