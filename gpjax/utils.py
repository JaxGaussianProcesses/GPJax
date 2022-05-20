import typing as tp
from copy import deepcopy
import jax
import jax.numpy as jnp

from .types import Array


def I(n: int) -> Array:
    """
    Compute an n x n identity matrix.
    :param n: The size of of the matrix.
    :return: An n x n identity matrix.
    """
    return jnp.eye(n)


def concat_dictionaries(a: dict, b: dict) -> dict:
    """
    Append one dictionary below another. If duplicate keys exist, then the key-value pair of the second supplied
    dictionary will be used.
    """
    return {**a, **b}


def merge_dictionaries(base_dict: dict, in_dict: dict) -> dict:
    """
    This will return a complete dictionary based on the keys of the first matrix. If the same key should exist in the
    second matrix, then the key-value pair from the first dictionary will be overwritten. The purpose of this is that
    the base_dict will be a complete dictionary of values such that an incomplete second dictionary can be used to
    update specific key-value pairs.

    :param base_dict: Complete dictionary of key-value pairs.
    :param in_dict: Subset of key-values pairs such that values from this dictionary will take precedent.
    :return: A merged single dictionary.
    """
    for k, v in base_dict.items():
        if k in in_dict.keys():
            base_dict[k] = in_dict[k]
    return base_dict


def sort_dictionary(base_dict: dict) -> dict:
    """
    Sort a dictionary based on the dictionary's key values.

    :param base_dict: The unsorted dictionary.
    :return: A dictionary sorted alphabetically on the dictionary's keys.
    """
    return dict(sorted(base_dict.items()))


def as_constant(parameter_set: dict, params: list) -> tp.Tuple[dict, dict]:
    base_params = deepcopy(parameter_set)
    sparams = {}
    for param in params:
        sparams[param] = base_params[param]
        del base_params[param]
    return base_params, sparams


def dict_array_coercion(params: tp.Dict) -> tp.Tuple[tp.Callable, tp.Callable]:
    """Construct the logic required to map a dictionary of parameters to an array of parameters. The values of the dictionary can themselves be dictionaries; the function should work recursively.

    Args:
        params (tp.Dict): The dictionary of parameters that we would like to map into an array.

    Returns:
        tp.Tuple[tp.Callable, tp.Callable]: A pair of functions, the first of which maps a dictionary to an array, and the second of which maps an array to a dictionary. The remapped dictionary is equal in structure to the original dictionary.
    """
    flattened_pytree = jax.tree_util.tree_flatten(params)

    def dict_to_array(parameter_dict) -> jnp.DeviceArray:
        return jax.tree_util.tree_flatten(parameter_dict)[0]

    def array_to_dict(parameter_array) -> tp.Dict:
        return jax.tree_util.tree_unflatten(flattened_pytree[1], parameter_array)

    return dict_to_array, array_to_dict
