from copy import deepcopy
import tensorflow as tf
import numpy as np
from .types import Kernel


def _broadcast_elementwise(func, a, b):
    flatres = func(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
    return tf.reshape(flatres, tf.concat([tf.shape(a), tf.shape(b)], 0))


def compute_distance_matrix(x, y):
    """
    For a given pair of 2-dimensional arrays, we here compute the Euclidean distance between all
    possible pairs of items. For reasons outlined in more detail [here](http://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf)
    this is a faster way to compute the distance matrix, although it can lead to some numerical
    instability.

    :param x: An [m x d] array
    :param y: An [n x d] array
    :return: An [m x n] array
    """
    if y is None:
        Xs = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        dist = -2 * tf.matmul(x, x, transpose_b=True)
        dist += Xs + tf.linalg.adjoint(Xs)
        return dist
    Xs = tf.reduce_sum(tf.square(x), axis=-1)
    Ys = tf.reduce_sum(tf.square(y), axis=-1)
    dist = -2 * tf.tensordot(x, y, [[-1], [-1]])
    dist += _broadcast_elementwise(tf.add, Xs, Ys)
    return dist


def compute_gram(kernel: Kernel, x: np.ndarray, y: np.ndarray = None, jitter: float = None):
    if y is None:
        y = deepcopy(x)
    D = compute_distance_matrix(x, y)
    gram = kernel.func(D)
    if jitter is not None:
        return stabilise(gram, jitter)
    else:
        return gram


def stabilise(A, jitter_amount: float = 1e-8):
    """
    Sometimes the smaller eigenvalues of a kernels' Gram matrix can be very slightly negative. This
    then leads to a non-invertible matrix. To account for this, we simply add a very small amount of
    noise, or jitter, to the Gram matrix's diagonal to _stabilise_ the eigenvalues.

    :param A: A, possibly non-invertible, Gram matrix
    :param jitter_amount: A tiny amount of noise to be summed onto the Gram matrix's diagonal.
    :return: A stabilised Gram matrix.
    """
    return A + tf.eye(A.shape[0], dtype=tf.float64) * jitter_amount
