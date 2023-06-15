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

from gpjax.linops.dense import Dense
from gpjax.linops.diagonal import Diagonal


def approx_equal(res: jax.Array, actual: jax.Array) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-6


# -----------------

# (1) Test initialisation


@pytest.mark.parametrize("n", [1, 2, 5])
def test_init(n: int) -> None:
    values = jr.uniform(_PRNGKey, (n,))
    diag = Diagonal(values)

    # Check types.
    assert isinstance(diag, Diagonal)
    assert is_dataclass(diag)

    # Check properties.
    assert diag.shape == (n, n)
    assert diag.dtype == jnp.float64
    assert diag.ndim == 2

    # Check pytree.
    for item1, item2 in zip(jtu.tree_leaves(diag), [values]):
        assert approx_equal(item1, item2)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag(n: int) -> None:
    entries = jr.uniform(_PRNGKey, (n,))
    diag = Diagonal(entries)
    res = diag.diagonal()
    actual = entries
    assert approx_equal(res, actual)


# (2) Test `to_dense` method


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    values = jr.uniform(_PRNGKey, (n,))
    diag = Diagonal(values)
    actual = jnp.diag(values)
    res = diag.to_dense()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_solve(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = Diagonal(values)
    rhs = jr.uniform(_PRNGKey, shape=(n,))

    assert approx_equal(diag.solve(rhs), rhs / values)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_inverse(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = Diagonal(values)
    res = diag.inverse()

    assert isinstance(res, Diagonal)
    assert approx_equal(res.to_dense(), jnp.diag(1 / diag.diagonal()))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_root(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = Diagonal(values)
    res = diag.to_root()
    actual = Diagonal(jnp.sqrt(values))

    assert isinstance(res, Diagonal)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_log_det(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = Diagonal(values)
    res = diag.log_det()
    actual = jnp.linalg.slogdet(jnp.diag(values))[1]

    approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_trace(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = Diagonal(values)
    res = diag.trace()
    actual = values.sum()

    assert res == actual


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_root(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    root = Diagonal(values)
    diag = Diagonal.from_root(root)
    res = diag.to_dense()
    actual = jnp.diag(root.diagonal() ** 2)

    assert isinstance(diag, Diagonal)
    assert approx_equal(res, actual)
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_dense(n: int) -> None:
    dense = jr.uniform(_PRNGKey, shape=(n, n))
    res = Diagonal.from_dense(dense)
    actual = jnp.diag(dense.diagonal())

    assert isinstance(res, Diagonal)
    assert approx_equal(res.to_dense(), actual)
    assert res.shape == actual.shape


# (3) Test add arithmetic:


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_array(n: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    array = jr.uniform(key1, (n, n))
    values = jr.uniform(key2, (n,))
    diag = Diagonal(values)

    res_left = diag + array
    res_right = array + diag
    actual = jnp.diag(values) + array

    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert approx_equal(res_left, actual)
    assert approx_equal(res_right, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_dense(n: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    array = jr.uniform(key1, (n, n))
    values = jr.uniform(key2, (n,))
    dense = Dense(array)
    diag = Diagonal(values)

    res_left = diag + dense
    res_right = dense + diag
    actual = jnp.diag(values) + array

    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert isinstance(res_left, Dense)
    assert isinstance(res_right, Dense)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_diagonal(n: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    values_a = jr.uniform(key1, (n,))
    values_b = jr.uniform(key2, (n,))
    diag_a = Diagonal(values_a)
    diag_b = Diagonal(values_b)

    res_left = diag_a + diag_b
    res_right = diag_b + diag_a

    actual = jnp.diag(values_a + values_b)

    assert approx_equal(res_left.to_dense(), actual)
    assert isinstance(res_left, Diagonal)
    assert approx_equal(res_right.to_dense(), actual)
    assert isinstance(res_right, Diagonal)


# (4) Test mul arithmetic:


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul_array(n: int) -> None:
    key, subkey = jr.split(_PRNGKey, 2)
    constant = jr.uniform(key, shape=())
    values = jr.uniform(subkey, shape=(n,))
    diag = Diagonal(values)

    res_left = diag * constant
    res_right = constant * diag

    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert approx_equal(res_left, constant * jnp.diag(values))
    assert approx_equal(res_right, constant * jnp.diag(values))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul_dense(n: int) -> None:
    key, subkey = jr.split(_PRNGKey, 2)
    array = jr.uniform(key, shape=(n, n))
    values = jr.uniform(subkey, shape=(n,))
    dense = Dense(array)
    diag = Diagonal(values)

    res_left = diag * dense
    res_right = dense * diag
    actual = jnp.diag(values) * array

    assert isinstance(res_left, Diagonal)
    assert isinstance(res_right, Diagonal)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul_diagonal(n: int) -> None:
    key, subkey = jr.split(_PRNGKey, 2)
    values_a = jr.uniform(key, shape=(n,))
    values_b = jr.uniform(subkey, shape=(n,))
    diag_a = Diagonal(values_a)
    diag_b = Diagonal(values_b)

    res_left = diag_a * diag_b
    res_right = diag_b * diag_a

    assert isinstance(res_left, Diagonal)
    assert isinstance(res_right, Diagonal)
    assert approx_equal(res_left.to_dense(), jnp.diag(values_a * values_b))
    assert approx_equal(res_right.to_dense(), jnp.diag(values_a * values_b))


# (5) Test matmul arithmetic:


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("p", [1, 2])
def test_matmul_array(n: int, m: int, p: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = Diagonal(values)

    array_left = jr.uniform(_PRNGKey, shape=(n, m))
    array_right = jr.uniform(_PRNGKey, shape=(m, n))

    res_left = diag @ array_left
    res_right = array_right @ diag
    actual_left = jnp.diag(values) @ array_left
    actual_right = array_right @ jnp.diag(values)

    assert res_left.shape == (n, m)
    assert res_right.shape == (m, n)
    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert approx_equal(res_left, actual_left)
    assert approx_equal(res_right, actual_right)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("m", [1, 2])
def test_matmul_dense(n: int, m: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = Diagonal(values)

    dense_left = Dense(jr.uniform(_PRNGKey, shape=(n, m)))
    dense_right = Dense(jr.uniform(_PRNGKey, shape=(m, n)))

    res_left = diag @ dense_left
    res_right = dense_right @ diag
    actual_left = jnp.diag(values) @ dense_left.matrix
    actual_right = dense_right.matrix @ jnp.diag(values)

    assert res_left.shape == (n, m)
    assert res_right.shape == (m, n)
    assert isinstance(res_left, Dense)
    assert isinstance(res_right, Dense)
    assert approx_equal(res_left.to_dense(), actual_left)
    assert approx_equal(res_right.to_dense(), actual_right)


@pytest.mark.parametrize("n", [1, 2])
def test_matmul_diag(n: int) -> None:
    key1, key2 = jr.split(_PRNGKey, 2)
    values1 = jr.uniform(key1, shape=(n,))
    values2 = jr.uniform(key2, shape=(n,))

    diag1 = Diagonal(values1)
    diag2 = Diagonal(values2)

    res_left = diag1 @ diag2
    res_right = diag2 @ diag1
    actual = jnp.diag(values1) @ jnp.diag(values2)

    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert isinstance(res_left, Diagonal)
    assert isinstance(res_right, Diagonal)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)
