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
from jaxlinop.zero_linear_operator import ZeroLinearOperator


def approx_equal(res: jax.Array, actual: jax.Array) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-6


@pytest.mark.parametrize("n", [1, 2, 5])
def test_init(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))
    assert zero.shape == (n, n)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))
    res = zero.diagonal()
    actual = jnp.zeros(n)
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))
    actual = jnp.zeros(shape=(n, n))
    res = zero.to_dense()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_diagonal(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))
    entries = jr.uniform(_PRNGKey, shape=(n,))
    diag = DiagonalLinearOperator(diag=entries)
    zero_add_diag = zero._add_diagonal(diag)

    assert isinstance(zero_add_diag, DiagonalLinearOperator)

    res = zero_add_diag.to_dense()
    actual = jnp.eye(n) * entries
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add(n: int) -> None:

    array = jr.uniform(_PRNGKey, shape=(n, n))
    entries = jr.uniform(_PRNGKey, shape=(n,))
    zero = ZeroLinearOperator(shape=(n, n))

    # Add array.
    res_left = zero + array
    res_right = array + zero

    assert isinstance(res_left, DenseLinearOperator)
    assert isinstance(res_right, DenseLinearOperator)

    # Add Dense.
    dense_lo = DenseLinearOperator(matrix=array)

    res_left = zero + dense_lo
    res_right = dense_lo + zero
    actual = dense_lo

    assert isinstance(res_left, DenseLinearOperator)
    assert isinstance(res_right, DenseLinearOperator)
    assert approx_equal(res_left.to_dense(), actual.to_dense())
    assert approx_equal(res_right.to_dense(), actual.to_dense())

    # Add Diagonal.
    diag = DiagonalLinearOperator(diag=entries)

    res_left = zero + diag
    res_right = diag + zero
    actual = diag

    assert isinstance(res_left, DiagonalLinearOperator)
    assert isinstance(res_right, DiagonalLinearOperator)
    assert approx_equal(res_left.to_dense(), actual.to_dense())
    assert approx_equal(res_right.to_dense(), actual.to_dense())


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul(n: int) -> None:
    constant = jr.uniform(_PRNGKey, shape=())
    zero = ZeroLinearOperator(shape=(n, n))

    res_left = zero * constant
    res_right = constant * zero

    assert isinstance(res_left, ZeroLinearOperator)
    assert isinstance(res_right, ZeroLinearOperator)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_matmul(n: int) -> None:
    array = jr.uniform(_PRNGKey, shape=(n, n))
    zero = ZeroLinearOperator(shape=(n, n))

    res_left = zero @ array
    res_right = array @ zero

    assert isinstance(res_left, ZeroLinearOperator)
    assert isinstance(res_right, ZeroLinearOperator)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_solve(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))

    with pytest.raises(RuntimeError):
        rhs = jr.uniform(_PRNGKey, shape=(n,))
        zero.solve(rhs)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_inverse(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))

    with pytest.raises(RuntimeError):
        zero.inverse()


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_root(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))

    res = zero.to_root()
    actual = zero

    assert isinstance(res, ZeroLinearOperator)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_log_det(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))

    assert zero.log_det() == jnp.log(0.0)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_trace(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))
    res = zero.trace()
    actual = 0.0

    assert res == actual


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_root(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))

    res = ZeroLinearOperator.from_root(zero)
    actual = zero

    assert isinstance(res, ZeroLinearOperator)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_dense(n: int) -> None:
    zero = ZeroLinearOperator(shape=(n, n))

    dense = jr.uniform(_PRNGKey, shape=(n, n))
    res = ZeroLinearOperator.from_dense(dense)
    actual = zero

    assert isinstance(res, ZeroLinearOperator)
    assert approx_equal(res.to_dense(), actual.to_dense())
    assert res.shape == actual.shape
