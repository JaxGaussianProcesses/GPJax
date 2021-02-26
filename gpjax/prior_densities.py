import jax.numpy as jnp
from jax.interpreters.ad import JVPTracer
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd

from .types import Array


@dispatch((jnp.DeviceArray, JVPTracer), tfd.Distribution)
def log_density(param: jnp.DeviceArray, density: tfd.Distribution) -> Array:
    return density.log_prob(param)
