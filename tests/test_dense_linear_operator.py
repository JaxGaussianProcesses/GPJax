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
import jax
import pytest
from jax.config import config


# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_PRNGKey = jr.PRNGKey(42)

from jaxlinop.diagonal_linear_operator import DiagonalLinearOperator
from jaxlinop.dense_linear_operator import DenseLinearOperator
from jaxlinop.triangular_linear_operator import LowerTriangularLinearOperator


def approx_equal(res: jax.Array, actual: jax.Array) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-6


@pytest.mark.parametrize("n", [1, 2, 5])
def test_init(n: int) -> None:
    values = jr.uniform(_PRNGKey, (n, n))
    dense = DenseLinearOperator(values)
    assert dense.shape == (n, n)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag(n: int) -> None:
    values = jr.uniform(_PRNGKey, (n, n))
    dense = DenseLinearOperator(values)
    res = dense.diagonal()
    actual = values.diagonal()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    values = jr.uniform(_PRNGKey, (n, n))
    dense = DenseLinearOperator(values)
    actual = values
    res = dense.to_dense()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_diagonal(n: int) -> None:

    # Test adding generic diagonal linear operator.
    key_a, key_b = jr.split(_PRNGKey)

    values_a = jr.uniform(key_a, (n, n))
    dense = DenseLinearOperator(values_a)

    values_b = jr.uniform(key_b, (n,))
    diag = DiagonalLinearOperator(values_b)

    res = dense._add_diagonal(diag)
    actual = values_a + jnp.diag(values_b)

    assert isinstance(res, DenseLinearOperator)
    assert res.shape == (n, n)
    assert approx_equal(res.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add(n: int) -> None:
    key = _PRNGKey

    array = jr.uniform(_PRNGKey, shape=(n, n))
    entries = jr.uniform(_PRNGKey, shape=(n,))
    values = jr.uniform(key, (n, n))
    dense = DenseLinearOperator(values)

    # Add array.
    res_left = dense + array
    res_right = array + dense

    assert approx_equal(res_left.to_dense(), array + values)
    assert approx_equal(res_right.to_dense(), array + values)
    assert isinstance(res_left, DenseLinearOperator)
    assert isinstance(res_right, DenseLinearOperator)

    # Add Dense.
    array = jr.uniform(_PRNGKey, shape=(n, n))
    dense = DenseLinearOperator(matrix=array)

    res_left = array + dense
    res_right = dense + array
    actual = dense.to_dense() + values

    assert isinstance(res_left, DenseLinearOperator)
    assert isinstance(res_right, DenseLinearOperator)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)

    # Add Diagonal.
    diag = DiagonalLinearOperator(diag=entries)

    res_left = dense + diag
    res_right = diag + dense
    actual = diag.to_dense() + values

    assert isinstance(res_left, DenseLinearOperator)
    assert isinstance(res_right, DenseLinearOperator)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul(n: int) -> None:
    key, subkey = jr.split(_PRNGKey, 2)
    constant = jr.uniform(key, shape=())
    values = jr.uniform(subkey, shape=(n, n))
    dense = DenseLinearOperator(values)

    res_left = dense * constant
    res_right = constant * dense

    assert isinstance(res_left, DenseLinearOperator)
    assert isinstance(res_right, DenseLinearOperator)
    assert approx_equal(res_left.to_dense(), values * constant)
    assert approx_equal(res_right.to_dense(), values * constant)


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_matmul(n: int, m: int) -> None:
    array_left = jr.uniform(_PRNGKey, shape=(n, m))
    array_right = jr.uniform(_PRNGKey, shape=(m, n))

    values = jr.uniform(_PRNGKey, shape=(n, n))
    values = values @ values.T
    dense = DenseLinearOperator(values)

    res_left = dense @ array_left
    res_right = array_right @ dense

    assert approx_equal(res_left, values @ array_left)
    assert approx_equal(res_right, array_right @ values)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_solve(n: int) -> None:
    sqrt = jr.uniform(_PRNGKey, shape=(n, n))
    values = sqrt @ sqrt.T
    dense = DenseLinearOperator(values)

    assert approx_equal(dense.solve(values), jnp.eye(n))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_inverse(n: int) -> None:
    sqrt = jr.uniform(_PRNGKey, shape=(n, n))
    values = sqrt @ sqrt.T
    dense = DenseLinearOperator(values)
    res = dense.inverse()

    assert isinstance(res, DenseLinearOperator)
    assert approx_equal(res.to_dense(), jnp.linalg.inv(values))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_root(n: int) -> None:
    sqrt = jr.uniform(_PRNGKey, shape=(n, n))
    values = sqrt @ sqrt.T
    dense = DenseLinearOperator(values)
    res = dense.to_root()
    actual = jnp.linalg.cholesky(values)

    assert isinstance(res, LowerTriangularLinearOperator)
    assert approx_equal(res.to_dense(), actual)
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_log_det(n: int) -> None:
    sqrt = jr.uniform(_PRNGKey, shape=(n, n))
    values = sqrt @ sqrt.T
    dense = DenseLinearOperator(values)
    res = dense.log_det()
    actual = jnp.linalg.slogdet(values)[1]

    approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_trace(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n, n))
    dense = DenseLinearOperator(values)
    res = dense.trace()
    actual = jnp.diag(values).sum()

    assert res == actual


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_root(n: int) -> None:
    sqrt = jr.uniform(_PRNGKey, shape=(n, n))
    values = sqrt @ sqrt.T
    L = jnp.linalg.cholesky(values)
    root = LowerTriangularLinearOperator.from_dense(L)
    dense = DenseLinearOperator.from_root(root)

    assert isinstance(dense, DenseLinearOperator)
    assert approx_equal(dense.to_root().to_dense(), root.to_dense())
    assert approx_equal(dense.to_dense(), values)
    assert root.shape == dense.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_dense(n: int) -> None:
    values = jr.uniform(_PRNGKey, shape=(n, n))
    res = DenseLinearOperator.from_dense(values)
    actual = DenseLinearOperator(values)

    assert isinstance(res, DenseLinearOperator)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape
