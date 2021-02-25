from typing import Union, Tuple
import jax.numpy as jnp
from jax.interpreters.pxla import ShardedDeviceArray
from chex import Array

# Array = Union[jnp.ndarray, ShardedDeviceArray, jnp.DeviceArray]  # Cannot currently dispatch on a Union type
# Data = Tuple[Array, Array]