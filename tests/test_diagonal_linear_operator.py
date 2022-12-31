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


import jax.numpy as jnp
import jax.random as jr
import pytest
import jax
from jax.config import config


# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_PRNGKey = jr.PRNGKey(42)

from jaxlinop.diagonal_linear_operator import DiagonalLinearOperator
from jaxlinop.dense_linear_operator import DenseLinearOperator


def approx_equal(res: jax.Array, actual: jax.Array) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-6


@pytest.mark.parametrize("n", [1, 2, 5])
def test_init(n: int) -> None:
    values = jr.uniform(_PRNGKey, (n,))
    diag = DiagonalLinearOperator(values)
    assert diag.shape == (n, n)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag(n: int) -> None:
    entries = jr.uniform(_PRNGKey, (n,))
    diag = DiagonalLinearOperator(entries)
    res = diag.diagonal()
    actual = entries
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    values = jr.uniform(_PRNGKey, (n,))
    diag = DiagonalLinearOperator(values)
    actual = jnp.diag(values)
    res = diag.to_dense()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_diagonal(n: int) -> None:

    # Test adding two diagonal linear operators.
    key_a, key_b = jr.split(_PRNGKey)

    values_a = jr.uniform(key_a, (n,))
    diag_a = DiagonalLinearOperator(values_a)

    values_b = jr.uniform(key_b, (n,))
    diag_b = DiagonalLinearOperator(values_b)

    res = diag_a._add_diagonal(diag_b)
    actual = jnp.diag(values_a + values_b)

    assert isinstance(res, DiagonalLinearOperator)
    assert res.shape == (n, n)
    assert approx_equal(res.to_dense(), actual)

    # Test adding on the generic diagonal linear operator.
    key = _PRNGKey

    values = jr.uniform(key, (n,))
    diag = DiagonalLinearOperator(values)
    actual = jnp.diag(values)

    random_diag = DiagonalLinearOperator(jr.uniform(key, (n,)))

    res = diag._add_diagonal(random_diag)
    actual = jnp.diag(values + random_diag.diagonal())

    assert isinstance(res, DiagonalLinearOperator)
    assert res.shape == (n, n)

    assert approx_equal(res.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add(n: int) -> None:
    key = _PRNGKey

    array = jr.uniform(_PRNGKey, shape=(n, n))
    entries = jr.uniform(_PRNGKey, shape=(n,))
    values = jr.uniform(key, (n,))
    diag = DiagonalLinearOperator(values)

    # Add array.
    res_left = diag + array
    res_right = array + diag

    assert approx_equal(res_left.to_dense(), array + jnp.diag(values))
    assert approx_equal(res_right.to_dense(), array + jnp.diag(values))

    # Add Dense.
    array = jr.uniform(_PRNGKey, shape=(n, n))
    dense = DenseLinearOperator(matrix=array)

    res_left = diag + dense
    res_right = dense + diag
    actual = dense.to_dense() + jnp.diag(values)

    assert isinstance(res_left, DenseLinearOperator)
    assert isinstance(res_right, DenseLinearOperator)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)

    # Add Diagonal.
    diag = DiagonalLinearOperator(diag=entries)

    res_left = diag + diag
    res_right = diag + diag
    actual = diag.to_dense() + jnp.diag(values)

    assert isinstance(res_left, DiagonalLinearOperator)
    assert isinstance(res_right, DiagonalLinearOperator)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul(n: int) -> None:
    key, subkey = jr.split(_PRNGKey, 2)
    constant = jr.uniform(key, shape=())
    values = jr.uniform(subkey, shape=(n,))
    diag = DiagonalLinearOperator(values)

    res_left = diag * constant
    res_right = constant * diag

    assert isinstance(res_left, DiagonalLinearOperator)
    assert isinstance(res_right, DiagonalLinearOperator)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_matmul(n: int) -> None:
    array = jr.uniform(_PRNGKey, shape=(n, n))
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = DiagonalLinearOperator(values)

    res_left = diag @ array
    res_right = array @ diag

    assert approx_equal(res_left, diag.to_dense() @ array)
    assert approx_equal(res_right, array @ diag.to_dense())


@pytest.mark.parametrize("n", [1, 2, 5])
def test_solve(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = DiagonalLinearOperator(values)
    rhs = jr.uniform(_PRNGKey, shape=(n,))
    diag.solve(rhs)

    assert approx_equal(diag.solve(rhs), rhs / values)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_inverse(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = DiagonalLinearOperator(values)
    res = diag.inverse()

    assert isinstance(res, DiagonalLinearOperator)
    assert approx_equal(res.to_dense(), jnp.diag(1 / diag.diagonal()))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_root(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = DiagonalLinearOperator(values)
    res = diag.to_root()
    actual = DiagonalLinearOperator(jnp.sqrt(values))

    assert isinstance(res, DiagonalLinearOperator)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_log_det(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = DiagonalLinearOperator(values)
    res = diag.log_det()
    actual = jnp.linalg.slogdet(jnp.diag(values))[1]

    approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_trace(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    diag = DiagonalLinearOperator(values)
    res = diag.trace()
    actual = values.sum()

    assert res == actual


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_root(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n,))
    root = DiagonalLinearOperator(values)
    diag = DiagonalLinearOperator.from_root(root)
    res = diag.to_dense()
    actual = jnp.diag(root.diagonal() ** 2)

    assert isinstance(diag, DiagonalLinearOperator)
    assert approx_equal(res, actual)
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_dense(n: int) -> None:
    dense = jr.uniform(_PRNGKey, shape=(n, n))
    res = DiagonalLinearOperator.from_dense(dense)
    actual = jnp.diag(dense.diagonal())

    assert isinstance(res, DiagonalLinearOperator)
    assert approx_equal(res.to_dense(), actual)
    assert res.shape == actual.shape
