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

import jax.numpy as jnp
import pytest
from jax.config import config

from gpjax.utils import (
    concat_dictionaries,
    dict_array_coercion,
    merge_dictionaries,
    sort_dictionary,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_concat_dict():
    d1 = {"a": 1, "b": 2}
    d2 = {"c": 3, "d": 4}
    d = concat_dictionaries(d1, d2)
    assert list(d.keys()) == ["a", "b", "c", "d"]
    assert list(d.values()) == [1, 2, 3, 4]


def test_merge_dicts():
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 3}
    d = merge_dictionaries(d1, d2)
    assert list(d.keys()) == ["a", "b"]
    assert list(d.values()) == [1, 3]


def test_sort_dict():
    unsorted = {"b": 1, "a": 2}
    sorted_dict = sort_dictionary(unsorted)
    assert list(sorted_dict.keys()) == ["a", "b"]
    assert list(sorted_dict.values()) == [2, 1]


@pytest.mark.parametrize("d", [1, 2, 10])
def test_array_coercion(d):
    params = {
        "kernel": {
            "lengthscale": jnp.array([1.0] * d),
            "variance": jnp.array([1.0]),
        },
        "likelihood": {"obs_noise": jnp.array([1.0])},
        "mean_function": {},
    }
    dict_to_array, array_to_dict = dict_array_coercion(params)
    assert array_to_dict(dict_to_array(params)) == params
    assert isinstance(dict_to_array(params), list)
    assert isinstance(array_to_dict(dict_to_array(params)), dict)
