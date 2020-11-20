import jax.numpy as jnp
from multipledispatch import dispatch
from .types import Zero


@dispatch(Zero, jnp.array)
def mean(mean_function, x):
    return jnp.zeros_like(x)
