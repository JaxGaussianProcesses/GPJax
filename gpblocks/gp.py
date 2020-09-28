import tensorflow as tf
import numpy as np


def marginal_log_likelihood(model, X, y):
    N = X.shape[0]
    Kxx = model.prior.kernel(X, X, jitter_amount=model.likelihood.obs_noise)
    L = tf.linalg.cholesky(Kxx)
    diag_sum = tf.reduce_sum(tf.linalg.diag_part(L))
    alpha = tf.linalg.triangular_solve(tf.transpose(L),
                                       tf.linalg.triangular_solve(L, y))
    return -0.5 * tf.squeeze(tf.matmul(
        tf.transpose(y), alpha)) - diag_sum - 0.5 * N * tf.math.log(
            2 * tf.cast(np.pi, dtype=tf.float64))


