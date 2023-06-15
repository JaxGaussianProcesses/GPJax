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
from gpjax.linops.triangular import LowerTriangular


def approx_equal(res: jax.Array, actual: jax.Array) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-6


# --- TODO: Move following to base class ---
@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag(n: int) -> None:
    values = jr.uniform(_PRNGKey, (n, n))
    dense = Dense(values)
    res = dense.diagonal()
    actual = values.diagonal()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_solve(n: int) -> None:
    sqrt = jr.uniform(_PRNGKey, shape=(n, n))
    values = sqrt @ sqrt.T
    dense = Dense(values)

    assert approx_equal(dense.solve(values), jnp.eye(n))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_inverse(n: int) -> None:
    sqrt = jr.uniform(_PRNGKey, shape=(n, n))
    values = sqrt @ sqrt.T
    dense = Dense(values)
    res = dense.inverse()

    assert isinstance(res, Dense)
    assert approx_equal(res.to_dense(), jnp.linalg.inv(values))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_root(n: int) -> None:
    sqrt = jr.uniform(_PRNGKey, shape=(n, n))
    values = sqrt @ sqrt.T
    dense = Dense(values)
    res = dense.to_root()
    actual = jnp.linalg.cholesky(values)

    assert isinstance(res, LowerTriangular)
    assert approx_equal(res.to_dense(), actual)
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_root(n: int) -> None:
    sqrt = jr.uniform(_PRNGKey, shape=(n, n))
    values = sqrt @ sqrt.T
    L = jnp.linalg.cholesky(values)
    root = LowerTriangular.from_dense(L)
    dense = Dense.from_root(root)

    assert isinstance(dense, Dense)
    assert approx_equal(dense.to_root().to_dense(), root.to_dense())
    assert approx_equal(dense.to_dense(), values)
    assert root.shape == dense.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_log_det(n: int) -> None:
    sqrt = jr.uniform(_PRNGKey, shape=(n, n))
    values = sqrt @ sqrt.T
    dense = Dense(values)
    res = dense.log_det()
    actual = jnp.linalg.slogdet(values)[1]
    approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_trace(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n, n))
    dense = Dense(values)
    res = dense.trace()
    actual = jnp.diag(values).sum()
    assert res == actual


# ------------------------------------------


# (1) Test initialisation


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_init(n: int, m: int) -> None:
    values = jr.uniform(_PRNGKey, (n, m))
    dense = Dense(values)

    # Check types.
    assert isinstance(dense, Dense)
    assert is_dataclass(dense)

    # Check properties.
    assert dense.shape == (n, m)
    assert dense.dtype == jnp.float64
    assert dense.ndim == 2

    # Check pytree.
    for item1, item2 in zip(jtu.tree_leaves(dense), [values]):
        assert approx_equal(item1, item2)


# (2) Test `to_dense` method


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    values = jr.uniform(_PRNGKey, (n, n))
    dense = Dense(values)
    actual = values
    res = dense.to_dense()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_dense(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n, n))
    res = Dense.from_dense(values)
    actual = Dense(values)

    assert isinstance(res, Dense)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape


# (3) Test add arithmetic:


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_add_array(n: int, m: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    array = jr.uniform(key1, (n, m))
    values = jr.uniform(key2, (n, m))
    dense = Dense(values)

    res_left = dense + array
    res_right = array + dense
    actual = values + array

    assert res_left.shape == (n, m)
    assert res_right.shape == (n, m)
    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert approx_equal(res_left, actual)
    assert approx_equal(res_right, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_add_dense(n: int, m: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    values1 = jr.uniform(key1, (n, m))
    values2 = jr.uniform(key2, (n, m))

    dense1 = Dense(values1)
    dense2 = Dense(values2)

    res_left = dense1 + dense2
    res_right = dense2 + dense1
    actual = values1 + values2

    assert res_left.shape == (n, m)
    assert res_right.shape == (n, m)
    assert isinstance(res_left, Dense)
    assert isinstance(res_right, Dense)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_diagonal(n: int) -> None:
    # Test adding generic diagonal linear operator.
    key1, key2 = jr.split(_PRNGKey)
    values1 = jr.uniform(key1, (n, n))
    values2 = jr.uniform(key2, (n,))
    dense = Dense(values1)
    diag = Diagonal(values2)

    res_left = dense + diag
    res_right = diag + dense
    actual = values1 + jnp.diag(values2)

    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert isinstance(res_left, Dense)
    assert isinstance(res_right, Dense)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_left.to_dense(), actual)


# (4) Test mul arithmetic:


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_mul_array(n: int, m: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    array = jr.uniform(key1, (n, m))
    values = jr.uniform(key2, (n, m))
    dense = Dense(values)

    res_left = dense * array
    res_right = array * dense
    actual = array * values

    assert res_left.shape == (n, m)
    assert res_right.shape == (n, m)
    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert approx_equal(res_left, actual)
    assert approx_equal(res_right, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_mul_dense(n: int, m: int) -> None:
    key1, key2 = jr.split(_PRNGKey)
    values1 = jr.uniform(key1, (n, m))
    values2 = jr.uniform(key2, (n, m))

    dense1 = Dense(values1)
    dense2 = Dense(values2)

    res_left = dense1 * dense2
    res_right = dense2 * dense1
    actual = values1 * values2

    assert res_left.shape == (n, m)
    assert res_right.shape == (n, m)
    assert isinstance(res_left, Dense)
    assert isinstance(res_right, Dense)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul_diagonal(n: int) -> None:
    # Test adding generic diagonal linear operator.
    key1, key2 = jr.split(_PRNGKey)
    values1 = jr.uniform(key1, (n, n))
    values2 = jr.uniform(key2, (n,))
    dense = Dense(values1)
    diag = Diagonal(values2)

    res_left = dense * diag
    res_right = diag * dense
    actual = values1 * jnp.diag(values2)

    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert isinstance(res_left, Diagonal)
    assert isinstance(res_right, Diagonal)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_left.to_dense(), actual)


# (5) Test matmul arithmetic:


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("p", [1, 2])
def test_matmul_array(n: int, m: int, p: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n, m))
    dense = Dense(values)

    array_left = jr.uniform(_PRNGKey, shape=(m, p))
    array_right = jr.uniform(_PRNGKey, shape=(p, n))

    actual_left = values @ array_left
    actual_right = array_right @ values

    res_left = dense @ array_left
    res_right = array_right @ dense

    assert res_left.shape == (n, p)
    assert res_right.shape == (p, m)
    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert approx_equal(res_left, actual_left)
    assert approx_equal(res_right, actual_right)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("p", [1, 2])
def test_matmul_dense(n: int, m: int, p: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n, m))
    dense = Dense(values)

    dense_left = Dense(jr.uniform(_PRNGKey, shape=(m, p)))
    dense_right = Dense(jr.uniform(_PRNGKey, shape=(p, n)))

    res_left = dense @ dense_left
    res_right = dense_right @ dense
    actual_left = values @ dense_left.matrix
    actual_right = dense_right.matrix @ values

    assert res_left.shape == (n, p)
    assert res_right.shape == (p, m)
    assert isinstance(res_left, Dense)
    assert isinstance(res_right, Dense)
    assert approx_equal(res_left.to_dense(), actual_left)
    assert approx_equal(res_right.to_dense(), actual_right)
