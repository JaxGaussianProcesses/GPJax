import jax.numpy as jnp


def scale(x: jnp.DeviceArray, factor: jnp.DeviceArray) -> jnp.DeviceArray:
    return x / factor


def stretch(K: jnp.DeviceArray, factor: jnp.DeviceArray) -> jnp.DeviceArray:
    return factor * K
