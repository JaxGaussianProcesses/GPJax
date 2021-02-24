import jax.numpy as jnp
from .types import Array


def I(n: int) -> Array:
    return jnp.eye(n)


def merge_dictionaries(a: dict, b:dict) -> dict:
    return {**a, **b}
