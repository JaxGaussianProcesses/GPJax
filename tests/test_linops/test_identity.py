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
from gpjax.linops.identity import Identity


def approx_equal(res: jax.Array, actual: jax.Array) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-6


# -----------------

# (1) Test initialisation


@pytest.mark.parametrize("n", [1, 2, 5])
def test_init(n: int) -> None:
    id = Identity(size=n)

    # Check types.
    assert isinstance(id, ConstantDiagonal)
    assert is_dataclass(id)

    # Check properties.
    assert id.shape == (n, n)
    assert id.dtype == jnp.float64
    assert id.ndim == 2
    assert id.size == n

    # Check pytree.
    assert jtu.tree_leaves(id) == [1.0]  # shape, dtype are static!


# (2) Test ops:


@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag(n: int) -> None:
    id = Identity(size=n)
    res = id.diagonal()
    actual = jnp.ones(n)
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    id = Identity(size=n)
    actual = jnp.eye(n)
    res = id.to_dense()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_solve(n: int, m: int) -> None:
    id = Identity(size=n)
    rhs = jr.uniform(_PRNGKey, shape=(n, m))
    res = id.solve(rhs)

    assert isinstance(res, jax.Array)
    assert approx_equal(res, rhs)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_root(n: int) -> None:
    id = Identity(size=n)
    res = id.to_root()
    assert isinstance(res, Identity)
    assert approx_equal(res.to_dense(), jnp.eye(n))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_root(n: int) -> None:
    id = Identity(size=n)
    res = Identity.from_root(id)
    assert isinstance(res, Identity)
    assert approx_equal(res.to_dense(), jnp.eye(n))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_dense(n: int) -> None:
    dense = jr.uniform(_PRNGKey, shape=(n, n))
    res = Identity.from_dense(dense)
    assert isinstance(res, Identity)
    assert approx_equal(res.to_dense(), jnp.eye(n))


# (3) Test add arithmetic:


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_array(n: int) -> None:
    id = Identity(size=n)
    array = jr.uniform(_PRNGKey, shape=(n, n))
    res = id + array
    actual = jnp.eye(n) + array

    assert isinstance(res, jax.Array)
    assert res.shape == (n, n)
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_dense(n: int) -> None:
    id = Identity(size=n)
    dense = jr.uniform(_PRNGKey, shape=(n, n))
    res = id + Dense(dense)
    actual = jnp.eye(n) + dense

    assert isinstance(res, Dense)
    assert res.shape == (n, n)
    assert approx_equal(res.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_diag(n: int) -> None:
    id = Identity(size=n)
    entries = jr.uniform(_PRNGKey, shape=(n,))
    diag = Diagonal(diag=entries)
    res = id + diag
    actual = jnp.eye(n) + jnp.diag(entries)

    assert isinstance(res, Diagonal)
    assert res.shape == (n, n)
    assert approx_equal(res.to_dense(), actual)


# (4) Test mul arithmetic:


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul_array(n: int) -> None:
    id = Identity(size=n)
    array = jr.uniform(_PRNGKey, shape=(n, n))
    res_left = id * array
    res_right = array * id
    actual = array * jnp.eye(n)

    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert approx_equal(res_left, actual)
    assert approx_equal(res_right, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul_dense(n: int) -> None:
    id = Identity(size=n)
    array = jr.uniform(_PRNGKey, shape=(n, n))
    dense = Dense(array)
    res_left = id * dense
    res_right = dense * id
    actual = array * jnp.eye(n)

    assert isinstance(res_left, Diagonal)
    assert isinstance(res_right, Diagonal)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul_diag(n: int) -> None:
    id = Identity(size=n)
    entries = jr.uniform(_PRNGKey, shape=(n,))
    diag = Diagonal(diag=entries)
    res_left = id * diag
    res_right = diag * id
    actual = jnp.diag(entries) * jnp.eye(n)

    assert isinstance(res_left, Diagonal)
    assert isinstance(res_right, Diagonal)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


# (5) Test matmul arithmetic:


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_matmul(n: int, m: int) -> None:
    array_right = jr.uniform(_PRNGKey, shape=(n, m))
    array_left = jr.uniform(_PRNGKey, shape=(m, n))
    id = Identity(size=n)
    res_left = id @ array_right
    res_right = array_left @ id
    actual_left = array_right
    actual_right = array_left

    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert approx_equal(res_left, actual_left)
    assert approx_equal(res_right, actual_right)


__all__ = [
    "Identity",
]
