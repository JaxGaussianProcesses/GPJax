from copy import deepcopy
import tensorflow as tf
import numpy as np
from .types import Kernel


def _broadcast_elementwise(func, a, b):
    flatres = func(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
    return tf.reshape(flatres, tf.concat([tf.shape(a), tf.shape(b)], 0))


def compute_distance_matrix(x, y):
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


def compute_gram(kernel: Kernel, x: np.ndarray, y: np.ndarray = None):
    if y is None:
        y = deepcopy(x)
    D = compute_distance_matrix(x, y)
    gram = kernel.func(D)
    return gram


def stabilise(A, jitter_amount: float = 1e-8):
    A + tf.eye(A.shape[0], dtype=tf.float64) * jitter_amount


# if __name__ == '__main__':
#     x = tf.reshape(tf.linspace(-1, 1, 100), (-1, 1))
#     kern = SquaredExponential(lengthscale=[0.1])
#     D = kern._distance(x, x)
#     assert D.shape[0] == D.shape[1]
#     assert D.shape[0] == x.shape[0]
#     assert tf.reduce_sum(tf.abs(tf.linalg.diag_part(D))) == 0.0
#     K = kern.compute_gram(x, x)
#     kern.lengthscale=[to_default_float(1.0)]
#     K2 = kern.compute_gram(x, x)
#     import matplotlib.pyplot as plt
#     fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
#     ax = axes.ravel()
#     ax[0].matshow(K)
#     ax[1].matshow(K2)
#     plt.show()
#     K3 = kern(x, x)
#     print(K3-K2)
