from gpblocks.gp import Prior, marginal_log_likelihood
from gpblocks.kernel import SquaredExponential
from gpblocks.likelihoods import Gaussian
from gpblocks.mean_functions import Zero
from gpblocks.samplers import sample
import tensorflow as tf


if __name__ == '__main__':
    x = tf.cast(tf.reshape(tf.linspace(-1., 1., 100), (-1, 1)),
                dtype=tf.float64)
    y = tf.sin(x) + tf.random.normal(x.shape, 0., 0.1, dtype=tf.float64)

    kern = SquaredExponential()
    lik = Gaussian()
    gp = Prior(Zero(), kern)

    posterior = gp * lik
    [print(p) for p in posterior.trainable_variables]
    mll = marginal_log_likelihood(posterior, x, y)
    print(mll)

    with tf.GradientTape() as tape:
        obj = marginal_log_likelihood(posterior, x, y)
    grad = tape.gradient(obj, posterior.trainable_variables)
    print(grad)
