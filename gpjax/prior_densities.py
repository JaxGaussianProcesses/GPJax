import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from multipledispatch import dispatch
from .types import Array
from jax.interpreters.ad import JVPTracer


@dispatch((jnp.DeviceArray, JVPTracer), tfd.Distribution)
def log_density(param: jnp.DeviceArray, density: tfd.Distribution) -> Array:
    return density.log_prob(param)
