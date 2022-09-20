import jax.numpy as jnp
import pytest

from gpjax.utils import (
    I,
    concat_dictionaries,
    dict_array_coercion,
    merge_dictionaries,
    sort_dictionary,
)


@pytest.mark.parametrize("n", [1, 10, 100])
def test_identity(n):
    identity = I(n)
    assert identity.shape == (n, n)
    assert (jnp.diag(identity) == jnp.ones(shape=(n,))).all()


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
