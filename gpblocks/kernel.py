import jax.numpy as jnp
from jax import vmap


def gram(kernel, xs):
    return vmap(lambda x: vmap(lambda y: kernel(x, y))(xs))(xs)


# TODO: Only works for 1-dimensional inputs right now
def rbf(lengthscale, variance):
    def kernel(x, y):
        tau = jnp.square(x-y)
        return jnp.square(variance)*jnp.exp(-tau/(2*jnp.square(lengthscale)))
    return kernel


def jitter_matrix(n: int, jitter_amount:float = 1e-6):
    return jnp.eye(n)*jitter_amount


def stabilise(A, jitter_amount: float = 1e-6):
    """
    Sometimes the smaller eigenvalues of a kernels' Gram matrix can be very slightly negative. This
    then leads to a non-invertible matrix. To account for this, we simply add a very small amount of
    noise, or jitter, to the Gram matrix's diagonal to _stabilise_ the eigenvalues.

    :param A: A, possibly non-invertible, Gram matrix
    :param jitter_amount: A tiny amount of noise to be summed onto the Gram matrix's diagonal.
    :return: A stabilised Gram matrix.
    """
    return A + jitter_matrix(A.shape[0], jitter_amount)
