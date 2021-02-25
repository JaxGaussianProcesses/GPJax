import jax.numpy as jnp
from .types import Array


def I(n: int) -> Array:
    return jnp.eye(n)


def concat_dictionaries(a: dict, b:dict) -> dict:
    return {**a, **b}


def merge_dictionaries(base_dict: dict, in_dict: dict) -> dict:
    for k, v in base_dict.items():
        if k in in_dict.keys():
            base_dict[k] = in_dict[k]
    return base_dict