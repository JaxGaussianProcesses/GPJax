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


from dataclasses import is_dataclass

import jax
from jax.config import config
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_PRNGKey = jr.PRNGKey(42)

from gpjax.linops.constant_diagonal import ConstantDiagonal
from gpjax.linops.dense import Dense
from gpjax.linops.diagonal import Diagonal


def approx_equal(res: jax.Array, actual: jax.Array) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-6


# -----------------

# (1) Test initialisation


@pytest.mark.parametrize("n", [1, 2, 5])
def test_init(n: int) -> None:
    value = jr.uniform(_PRNGKey, (1,))
    constant_diag = ConstantDiagonal(value=value, size=n)

    # Check types.
    assert isinstance(constant_diag, ConstantDiagonal)
    assert is_dataclass(constant_diag)

    # Check properties.
    assert constant_diag.shape == (n, n)
    assert constant_diag.dtype == jnp.float64
    assert constant_diag.ndim == 2
    assert constant_diag.size == n

    # Check pytree.
    assert jtu.tree_leaves(constant_diag) == [value]  # shape, dtype are static!


@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag(n: int) -> None:
    value = jr.uniform(_PRNGKey, (1,))
    constant_diag = ConstantDiagonal(value=value, size=n)
    res = constant_diag.diagonal()
    actual = jnp.ones(n) * value
    assert approx_equal(res, actual)


# (2) Test `to_dense` method


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    value = jr.uniform(_PRNGKey, (1,))
    constant_diag = ConstantDiagonal(value=value, size=n)
    actual = jnp.diag(jnp.ones(n) * value)
    res = constant_diag.to_dense()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_solve(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonal(value=value, size=n)
    rhs = jr.uniform(_PRNGKey, shape=(n,))
    constant_diag.solve(rhs)

    assert approx_equal(constant_diag.solve(rhs).to_dense(), rhs / value)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_inverse(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonal(value=value, size=n)

    res = constant_diag.inverse()

    assert isinstance(res, ConstantDiagonal)
    assert approx_equal(res.to_dense(), jnp.diag(1 / constant_diag.diagonal()))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_root(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonal(value=value, size=n)

    res = constant_diag.to_root()
    actual = ConstantDiagonal(value=jnp.sqrt(value), size=n)

    assert isinstance(res, ConstantDiagonal)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_log_det(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonal(value=value, size=n)
    res = constant_diag.log_det()
    actual = jnp.log(value) * n

    approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_trace(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonal(value=value, size=n)
    res = constant_diag.trace()
    actual = value * n

    assert res == actual


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_root(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    root = ConstantDiagonal(value=value, size=n)
    constant_diag = ConstantDiagonal.from_root(root)
    res = constant_diag.to_dense()
    actual = jnp.diag(root.diagonal() ** 2)

    assert isinstance(constant_diag, ConstantDiagonal)
    assert approx_equal(res, actual)
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_dense(n: int) -> None:
    dense = jr.uniform(_PRNGKey, shape=(n, n))
    res = ConstantDiagonal.from_dense(dense)
    actual = jnp.diag(jnp.ones(n) * dense[0, 0])

    assert isinstance(res, ConstantDiagonal)
    assert approx_equal(res.to_dense(), actual)
    assert res.shape == actual.shape


# (3) Test add arithmetic:


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_array(n: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    array = jr.uniform(key1, shape=(n, n))
    value = jr.uniform(key2, (1,))
    constant_diag = ConstantDiagonal(value=value, size=n)

    res_left = constant_diag + array
    res_right = array + constant_diag
    actual = jnp.diag(jnp.ones(n) * value) + array

    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert approx_equal(res_left, actual)
    assert approx_equal(res_right, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_dense(n: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    array = jr.uniform(key1, shape=(n, n))
    value = jr.uniform(key2, (1,))
    dense = Dense(array)
    constant_diag = ConstantDiagonal(value=value, size=n)

    res_left = constant_diag + dense
    res_right = dense + constant_diag
    actual = jnp.diag(jnp.ones(n) * value) + array

    assert isinstance(res_left, Dense)
    assert isinstance(res_right, Dense)
    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_diagonal(n: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    entries = jr.uniform(key1, shape=(n,))
    value = jr.uniform(key2, (1,))
    constant_diag = ConstantDiagonal(value=value, size=n)
    diag = Diagonal(entries)

    res_left = constant_diag + diag
    res_right = diag + constant_diag
    actual = jnp.diag(jnp.ones(n) * value + entries)

    assert isinstance(res_left, Diagonal)
    assert isinstance(res_right, Diagonal)
    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_constant_diag(n: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    value1 = jr.uniform(key1, shape=(1,))
    value2 = jr.uniform(key2, (1,))
    constant_diag1 = ConstantDiagonal(value=value1, size=n)
    constant_diag2 = ConstantDiagonal(value=value2, size=n)

    res_left = constant_diag1 + constant_diag2
    res_right = constant_diag2 + constant_diag1
    actual = jnp.diag(jnp.ones(n) * (value1 + value2))

    assert isinstance(res_left, ConstantDiagonal)
    assert isinstance(res_right, ConstantDiagonal)
    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


# (4) Test mul arithmetic:


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul(n: int) -> None:
    key, subkey = jr.split(_PRNGKey, 2)
    constant = jr.uniform(key, shape=())
    value = jr.uniform(subkey, shape=(1,))
    constant_diag = ConstantDiagonal(value=value, size=n)

    res_left = constant_diag * constant
    res_right = constant * constant_diag

    assert isinstance(res_left, Diagonal)
    assert isinstance(res_right, Diagonal)
    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert approx_equal(res_left.to_dense(), constant * constant_diag.to_dense())
    assert approx_equal(res_right.to_dense(), constant_diag.to_dense() * constant)


# (5) Test matmul arithmetic:


@pytest.mark.parametrize("n", [1, 2, 5])
def test_matmul(n: int) -> None:
    array = jr.uniform(_PRNGKey, shape=(n, n))
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonal(value=value, size=n)

    res_left = constant_diag @ array
    res_right = array @ constant_diag

    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert approx_equal(res_left, constant_diag.to_dense() @ array)
    assert approx_equal(res_right, array @ constant_diag.to_dense())
