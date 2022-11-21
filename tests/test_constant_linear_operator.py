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
from jax.config import config


# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_PRNGKey = jr.PRNGKey(42)

from jaxlinop.diagonal_linear_operator import DiagonalLinearOperator
from jaxlinop.dense_linear_operator import DenseLinearOperator
from jaxlinop.constant_diagonal_linear_operator import ConstantDiagonalLinearOperator


def approx_equal(res: jnp.ndarray, actual: jnp.ndarray) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-6


@pytest.mark.parametrize("n", [1, 2, 5])
def test_init(n: int) -> None:
    value = jr.uniform(_PRNGKey, (1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)
    assert constant_diag.shape == (n, n)
    assert constant_diag.size == n


@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag(n: int) -> None:
    value = jr.uniform(_PRNGKey, (1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)
    res = constant_diag.diagonal()
    actual = jnp.ones(n) * value
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    value = jr.uniform(_PRNGKey, (1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)
    actual = jnp.diag(jnp.ones(n) * value)
    res = constant_diag.to_dense()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_diagonal(n: int) -> None:

    # Test adding two constant diagonal linear operators.
    key_a, key_b = jr.split(_PRNGKey)

    value_a = jr.uniform(key_a, (1,))
    constant_diag_a = ConstantDiagonalLinearOperator(value=value_a, size=n)

    value_b = jr.uniform(key_b, (1,))
    constant_diag_b = ConstantDiagonalLinearOperator(value=value_b, size=n)

    res = constant_diag_a._add_diagonal(constant_diag_b)
    actual = jnp.diag(jnp.ones(n) * (value_a + value_b))

    assert isinstance(res, ConstantDiagonalLinearOperator)
    assert res.shape == (n, n)
    assert res.size == n
    assert approx_equal(res.to_dense(), actual)

    # Test adding on the generic diagonal linear operator.
    key = _PRNGKey

    value = jr.uniform(key, (1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)
    actual = jnp.diag(jnp.ones(n) * value)

    random_diag = DiagonalLinearOperator(jr.uniform(key, (n,)))

    res = constant_diag._add_diagonal(random_diag)
    actual = jnp.diag(jnp.ones(n) * value + random_diag.diagonal())

    assert isinstance(res, DiagonalLinearOperator)
    assert res.shape == (n, n)

    assert approx_equal(res.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add(n: int) -> None:
    key = _PRNGKey

    array = jr.uniform(_PRNGKey, shape=(n, n))
    entries = jr.uniform(_PRNGKey, shape=(n,))
    value = jr.uniform(key, (1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)

    # Add array.
    res_left = constant_diag + array
    res_right = array + constant_diag

    assert approx_equal(res_left.to_dense(), array + value * jnp.eye(n))
    assert approx_equal(res_right.to_dense(), array + value * jnp.eye(n))

    # Add Dense.
    array = jr.uniform(_PRNGKey, shape=(n, n))
    dense = DenseLinearOperator(matrix=array)

    res_left = constant_diag + dense
    res_right = dense + constant_diag
    actual = dense.to_dense() + value * jnp.eye(n)

    assert isinstance(res_left, DenseLinearOperator)
    assert isinstance(res_right, DenseLinearOperator)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)

    # Add Diagonal.
    diag = DiagonalLinearOperator(diag=entries)

    res_left = constant_diag + diag
    res_right = diag + constant_diag
    actual = diag.to_dense() + value * jnp.eye(n)

    assert isinstance(res_left, DiagonalLinearOperator)
    assert isinstance(res_right, DiagonalLinearOperator)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul(n: int) -> None:
    key, subkey = jr.split(_PRNGKey, 2)
    constant = jr.uniform(key, shape=())
    value = jr.uniform(subkey, shape=(1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)

    res_left = constant_diag * constant
    res_right = constant * constant_diag

    assert isinstance(res_left, ConstantDiagonalLinearOperator)
    assert isinstance(res_right, ConstantDiagonalLinearOperator)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_matmul(n: int) -> None:
    array = jr.uniform(_PRNGKey, shape=(n, n))
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)

    res_left = constant_diag @ array
    res_right = array @ constant_diag

    assert approx_equal(res_left, constant_diag.to_dense() @ array)
    assert approx_equal(res_right, array @ constant_diag.to_dense())


@pytest.mark.parametrize("n", [1, 2, 5])
def test_solve(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)
    rhs = jr.uniform(_PRNGKey, shape=(n,))
    constant_diag.solve(rhs)

    assert approx_equal(constant_diag.solve(rhs), rhs / value)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_inverse(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)

    res = constant_diag.inverse()

    assert isinstance(res, ConstantDiagonalLinearOperator)
    assert approx_equal(res.to_dense(), jnp.diag(1 / constant_diag.diagonal()))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_root(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)

    res = constant_diag.to_root()
    actual = ConstantDiagonalLinearOperator(value=jnp.sqrt(value), size=n)

    assert isinstance(res, ConstantDiagonalLinearOperator)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_log_det(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)
    res = constant_diag.log_det()
    actual = jnp.log(value) * n

    approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_trace(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    constant_diag = ConstantDiagonalLinearOperator(value=value, size=n)
    res = constant_diag.trace()
    actual = value * n

    assert res == actual


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_root(n: int) -> None:
    value = jr.uniform(_PRNGKey, shape=(1,))
    root = ConstantDiagonalLinearOperator(value=value, size=n)
    constant_diag = ConstantDiagonalLinearOperator.from_root(root)
    res = constant_diag.to_dense()
    actual = jnp.diag(root.diagonal() ** 2)

    assert isinstance(constant_diag, ConstantDiagonalLinearOperator)
    assert approx_equal(res, actual)
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_dense(n: int) -> None:
    dense = jr.uniform(_PRNGKey, shape=(n, n))
    res = ConstantDiagonalLinearOperator.from_dense(dense)
    actual = jnp.diag(jnp.ones(n) * dense[0, 0])

    assert isinstance(res, ConstantDiagonalLinearOperator)
    assert approx_equal(res.to_dense(), actual)
    assert res.shape == actual.shape
