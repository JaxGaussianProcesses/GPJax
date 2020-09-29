import tensorflow as tf
import numpy as np
from .types import Posterior


def marginal_log_likelihood(model: Posterior, X, y):
    """
    Compute the marginal log-likelihood of a GP with Gaussian likelihood

    :param model: A GP posterior struct
    :param X: Some observerved 2-dimensional inputs
    :param y: Observed 2-dimensional outputs
    :return: A floating point value of the current marginal log-likelihood
    """
    N = X.shape[0]
    Kxx = model.prior.kernel(X, X, jitter_amount=model.likelihood.obs_noise)
    L = tf.linalg.cholesky(Kxx)
    diag_sum = tf.reduce_sum(tf.linalg.diag_part(L))
    alpha = tf.linalg.triangular_solve(tf.transpose(L),
                                       tf.linalg.triangular_solve(L, y))
    return -0.5 * tf.squeeze(tf.matmul(
        tf.transpose(y), alpha)) - diag_sum - 0.5 * N * tf.math.log(
            2 * tf.cast(np.pi, dtype=tf.float64))


