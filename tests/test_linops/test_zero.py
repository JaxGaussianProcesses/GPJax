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
from gpjax.linops.zero import Zero


def approx_equal(res: jax.Array, actual: jax.Array) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-6


# -----------------

# (1) Test initialisation


@pytest.mark.parametrize("n", [1, 2, 5])
def test_init(n: int) -> None:
    zero = Zero(shape=(n, n))

    # Check types.
    assert isinstance(zero, Zero)
    assert is_dataclass(zero)

    # Check properties.
    assert zero.shape == (n, n)
    assert zero.dtype == jnp.float64
    assert zero.ndim == 2

    # Check pytree.
    assert jtu.tree_leaves(zero) == []  # shape, dtype are static!


# (2) Test ops


@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag(n: int) -> None:
    zero = Zero(shape=(n, n))
    res = zero.diagonal()
    actual = jnp.zeros(n)
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    zero = Zero(shape=(n, n))
    actual = jnp.zeros(shape=(n, n))
    res = zero.to_dense()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_solve(n: int) -> None:
    zero = Zero(shape=(n, n))

    with pytest.raises(RuntimeError):
        rhs = jr.uniform(_PRNGKey, shape=(n,))
        zero.solve(rhs)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_inverse(n: int) -> None:
    zero = Zero(shape=(n, n))

    with pytest.raises(RuntimeError):
        zero.inverse()


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_root(n: int) -> None:
    zero = Zero(shape=(n, n))

    res = zero.to_root()
    actual = zero

    assert isinstance(res, Zero)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_log_det(n: int) -> None:
    zero = Zero(shape=(n, n))

    assert zero.log_det() == jnp.log(0.0)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_trace(n: int) -> None:
    zero = Zero(shape=(n, n))
    res = zero.trace()
    actual = 0.0

    assert res == actual


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_root(n: int) -> None:
    zero = Zero(shape=(n, n))

    res = Zero.from_root(zero)
    actual = zero

    assert isinstance(res, Zero)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_dense(n: int) -> None:
    zero = Zero(shape=(n, n))

    dense = jr.uniform(_PRNGKey, shape=(n, n))
    res = Zero.from_dense(dense)
    actual = zero

    assert isinstance(res, Zero)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape


# (3) Test add arithmetic


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_add_array(n: int, m: int) -> None:
    zero = Zero(shape=(n, m))
    array = jr.uniform(_PRNGKey, shape=(n, m))

    res_left = zero + array
    res_right = array + zero

    actual = array

    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert res_left.shape == (n, m)
    assert res_right.shape == (n, m)
    assert approx_equal(res_left, actual)
    assert approx_equal(res_right, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_add_dense(n: int, m: int) -> None:
    zero = Zero(shape=(n, m))
    array = jr.uniform(_PRNGKey, shape=(n, m))
    dense = Dense.from_dense(array)

    res_left = zero + dense
    res_right = dense + zero

    actual = array

    assert isinstance(res_left, Dense)
    assert isinstance(res_right, Dense)
    assert res_left.shape == (n, m)
    assert res_right.shape == (n, m)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_diagonal(n: int) -> None:
    zero = Zero(shape=(n, n))
    entries = jr.uniform(_PRNGKey, shape=(n,))
    diag = Diagonal(diag=entries)

    res_left = zero + diag
    res_right = diag + zero

    actual = jnp.diag(entries)

    assert isinstance(res_left, Diagonal)
    assert isinstance(res_right, Diagonal)
    assert res_left.shape == (n, n)
    assert res_right.shape == (n, n)
    assert approx_equal(res_left.to_dense(), actual)


# (4) Test mul arithmetic


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_mul_constant(n: int, m: int) -> None:
    constant = jr.uniform(_PRNGKey, shape=())
    zero = Zero(shape=(n, m))

    res_left = zero * constant
    res_right = constant * zero

    actual = jnp.zeros((n, m))

    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert res_left.shape == (n, m)
    assert res_right.shape == (n, m)
    assert approx_equal(res_left, actual)
    assert approx_equal(res_right, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_mul_array(n: int, m: int) -> None:
    array = jr.uniform(_PRNGKey, shape=(n, m))
    zero = Zero(shape=(n, m))

    res_left = zero * array
    res_right = array * zero

    actual = jnp.zeros((n, m))

    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert res_left.shape == (n, m)
    assert res_right.shape == (n, m)
    assert approx_equal(res_left, actual)
    assert approx_equal(res_right, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_mul_dense(n: int, m: int) -> None:
    array = jr.uniform(_PRNGKey, shape=(n, m))
    dense = Dense.from_dense(array)
    zero = Zero(shape=(n, m))

    res_left = zero * dense
    res_right = dense * zero

    actual = jnp.zeros((n, m))

    assert isinstance(res_left, Zero)
    assert isinstance(res_right, Zero)
    assert res_left.shape == (n, m)
    assert res_right.shape == (n, m)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


# (5) Test matmul arithmetic


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("p", [1, 2])
def test_matmul_array(n: int, m: int, p: int) -> None:
    zero = Zero(shape=(n, m))

    array_left = jr.uniform(_PRNGKey, shape=(m, p))
    array_right = jr.uniform(_PRNGKey, shape=(p, n))

    actual_left = jnp.zeros((n, p))
    actual_right = jnp.zeros((p, m))

    res_left = zero @ array_left
    res_right = array_right @ zero

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
    zero = Zero(shape=(n, m))

    dense_left = Dense(jr.uniform(_PRNGKey, shape=(m, p)))
    dense_right = Dense(jr.uniform(_PRNGKey, shape=(p, n)))

    res_left = zero @ dense_left
    res_right = dense_right @ zero
    actual_left = jnp.zeros((n, p))
    actual_right = jnp.zeros((p, m))

    assert res_left.shape == (n, p)
    assert res_right.shape == (p, m)
    assert isinstance(res_left, Zero)
    assert isinstance(res_right, Zero)
    assert approx_equal(res_left.to_dense(), actual_left)
    assert approx_equal(res_right.to_dense(), actual_right)
