from typing import Tuple, Union

import jax.numpy as jnp
from chex import Array
from jax.interpreters.pxla import ShardedDeviceArray

# Array = Union[jnp.ndarray, ShardedDeviceArray, jnp.DeviceArray]  # Cannot currently dispatch on a Union type
# Data = Tuple[Array, Array]
NoneType = type(None)
