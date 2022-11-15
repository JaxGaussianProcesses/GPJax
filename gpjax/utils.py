# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp


def concat_dictionaries(a: Dict, b: Dict) -> Dict:
    """
    Append one dictionary below another. If duplicate keys exist, then the
    key-value pair of the second supplied dictionary will be used.

    Args:
        a (Dict): The first dictionary.
        b (Dict): The second dictionary.

    Returns:
        Dict: The merged dictionary.
    """
    return {**a, **b}


def merge_dictionaries(base_dict: Dict, in_dict: Dict) -> Dict:
    """
    This will return a complete dictionary based on the keys of the first
    matrix. If the same key should exist in the second matrix, then the
    key-value pair from the first dictionary will be overwritten. The purpose of
    this is that the base_dict will be a complete dictionary of values such that
    an incomplete second dictionary can be used to update specific key-value
    pairs.

    Args:
        base_dict (Dict): Complete dictionary of key-value pairs.
        in_dict (Dict): Subset of key-values pairs such that values from this
            dictionary will take precedent.

    Returns:
        Dict: A dictionary with the same keys as the base_dict, but with
            values from the in_dict.
    """
    for k, _ in base_dict.items():
        if k in in_dict.keys():
            base_dict[k] = in_dict[k]
    return base_dict


def sort_dictionary(base_dict: Dict) -> Dict:
    """
    Sort a dictionary based on the dictionary's key values.

    Args:
        base_dict (Dict): The dictionary to be sorted.

    Returns:
        Dict: The dictionary sorted alphabetically on the dictionary's keys.
    """
    return dict(sorted(base_dict.items()))


def dict_array_coercion(params: Dict) -> Tuple[Callable, Callable]:
    """
    Construct the logic required to map a dictionary of parameters to an array
    of parameters. The values of the dictionary can themselves be dictionaries;
    the function should work recursively.

    Args:
        params (Dict): The dictionary of parameters that we would like to map
            into an array.

    Returns:
        Tuple[Callable, Callable]: A pair of functions, the first of which maps
            a dictionary to an array, and the second of which maps an array to a
            dictionary. The remapped dictionary is equal in structure to the original
            dictionary.
    """
    flattened_pytree = jax.tree_util.tree_flatten(params)

    def dict_to_array(parameter_dict) -> jnp.DeviceArray:
        return jax.tree_util.tree_flatten(parameter_dict)[0]

    def array_to_dict(parameter_array) -> Dict:
        return jax.tree_util.tree_unflatten(flattened_pytree[1], parameter_array)

    return dict_to_array, array_to_dict


__all__ = [
    "concat_dictionaries",
    "merge_dictionaries",
    "sort_dictionary",
    "dict_array_coercion",
]
