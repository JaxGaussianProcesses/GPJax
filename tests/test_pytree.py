# Copyright 2022 The JaxLinOp Contributors. All Rights Reserved.
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

"""Adapted from Distrax._src.utils.test_jittable."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from typing import Any
from jaxlinop import pytree


class DummyJittable(pytree.Pytree):
    def __init__(self, params):
        self.name = "dummy"  # Non-JAX property, cannot be traced.
        self.data = {"params": params}  # Tree property, must be traced recursively.


def test_jittable():
    @jax.jit
    def get_params(obj):
        return obj.data["params"]

    obj = DummyJittable(jnp.ones((5,)))
    np.testing.assert_array_equal(get_params(obj), obj.data["params"])


def test_vmappable():
    def do_sum(obj):
        return obj.data["params"].sum()

    obj = DummyJittable(jnp.array([[1, 2, 3], [4, 5, 6]]))

    np.testing.assert_array_equal(do_sum(obj), obj.data["params"].sum())

    np.testing.assert_array_equal(
        jax.vmap(do_sum, in_axes=0)(obj), obj.data["params"].sum(axis=1)
    )

    np.testing.assert_array_equal(
        jax.vmap(do_sum, in_axes=1)(obj), obj.data["params"].sum(axis=0)
    )


def test_traceable():
    @jax.jit
    def inner_fn(obj):
        obj.data["params"] *= 3  # Modification after passing to jitted fn.
        return obj.data["params"].sum()

    def loss_fn(params):
        obj = DummyJittable(params)
        obj.data["params"] *= 2  # Modification before passing to jitted fn.
        return inner_fn(obj)

    params = np.ones((5,))
    # Both modifications will be traced if data tree is correctly traversed.
    grad_expected = params * 2 * 3
    grad = jax.grad(loss_fn)(params)
    np.testing.assert_array_equal(grad, grad_expected)

    params = jnp.ones((5,))
    # Both modifications will be traced if data tree is correctly traversed.
    grad_expected = params * 2 * 3
    grad = jax.grad(loss_fn)(params)
    np.testing.assert_array_equal(grad, grad_expected)


def test_different_jittables_to_compiled_function():
    @jax.jit
    def add_one_to_params(obj):
        obj.data["params"] = obj.data["params"] + 1
        return obj

    add_one_to_params(DummyJittable(np.zeros((5,))))
    add_one_to_params(DummyJittable(np.ones((5,))))

    add_one_to_params(DummyJittable(jnp.zeros((5,))))
    add_one_to_params(DummyJittable(jnp.ones((5,))))


def test_modifying_object_data_does_not_leak_tracers():
    @jax.jit
    def add_one_to_params(obj):
        obj.data["params"] = obj.data["params"] + 1
        return obj

    dummy = DummyJittable(jnp.ones((5,)))
    dummy_out = add_one_to_params(dummy)
    dummy_out.data["params"] -= 1


def test_metadata_modification_statements_are_removed_by_compilation():
    @jax.jit
    def add_char_to_name(obj):
        obj.name += "_x"
        return obj

    dummy = DummyJittable(jnp.ones((5,)))
    dummy_out = add_char_to_name(dummy)
    dummy_out = add_char_to_name(dummy)  # `name` change has been compiled out.
    dummy_out.name += "y"
    assert dummy_out.name == "dummy_xy"


@pytest.mark.parametrize("x", [1, 1.0, True, None])
def test_is_jax_type(x: Any) -> None:
    assert pytree.is_jax_type(x) == False
