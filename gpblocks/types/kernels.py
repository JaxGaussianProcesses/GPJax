import tensorflow as tf
from ..utils import to_default_float
from gpflow.base import Module, Parameter


# TODO: Slim down kernels and move functions like k and stabilise into kernels.py
class Kernel(Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, X, Y, jitter_amount=0):
        K = self.compute_gram(X, Y)
        return self.stabilise(K, jitter_amount)

    def _k(self, D):
        raise NotImplementedError

    def compute_gram(self, X, Y):
        raise NotImplementedError

    def stabilise(self, X, jitter_amount):
        if X.shape[0] == X.shape[1]:
            return X + tf.eye(X.shape[0], dtype=tf.float64) * jitter_amount


class Stationary(Kernel):
    def __init__(self, lengthscale=[1.0], variance=1.0, name=None):
        super().__init__(name=name)
        self.lengthscale = Parameter([to_default_float(l) for l in lengthscale])
        self.variance = Parameter(to_default_float(variance))

    @staticmethod
    def broadcasting_elementwise(op, a, b):
        """
        Apply binary operation `op` to every pair in tensors `a` and `b`.
        :param op: binary operator on tensors, e.g. tf.add, tf.substract
        :param a: tf.Tensor, shape [n_1, ..., n_a]
        :param b: tf.Tensor, shape [m_1, ..., m_b]
        :return: tf.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
        """
        flatres = op(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
        return tf.reshape(flatres, tf.concat([tf.shape(a), tf.shape(b)], 0))

    def _distance(self, X, Y):
        """
        Compute the pariwise Euclidean distance between two matrices A and B.
        Assuming A is an [N x D] matrix and B is [M x D], the resultant distance matrix will be [N x M].
        Result from http://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
        """
        if Y is None:
            Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += Xs + tf.linalg.adjoint(Xs)
            return dist
        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        Ys = tf.reduce_sum(tf.square(Y), axis=-1)
        dist = -2 * tf.tensordot(X, Y, [[-1], [-1]])
        dist += self.broadcasting_elementwise(tf.add, Xs, Ys)
        return dist


class SquaredExponential(Stationary):
    def __init__(self, lengthscale=[1.0], variance=[1.0]):
        super().__init__(lengthscale=lengthscale,
                         variance=variance,
                         name="SquaredExponential")

    def _k(self, A):
        return tf.square(self.variance) * tf.exp(
            -A / (2 * tf.square(self.lengthscale)))

    def compute_gram(self, X, Y):
        D = self._distance(X, Y)
        return self._k(D)
