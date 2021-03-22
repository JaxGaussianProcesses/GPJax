import jax.numpy as jnp
from jax.interpreters.pxla import ShardedDeviceArray
from jax.interpreters.ad import JVPTracer
from jax.interpreters.partial_eval import DynamicJaxprTracer


NoneType = type(None)
Array = (jnp.ndarray, ShardedDeviceArray, jnp.DeviceArray, JVPTracer, DynamicJaxprTracer)