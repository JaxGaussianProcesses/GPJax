import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Kernel:
    pass


@struct.dataclass
class Stationary(Kernel):
    lengthscale: jnp.array = jnp.array([1.0])
    variance: jnp.array = jnp.array([1.0])
