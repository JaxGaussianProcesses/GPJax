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
from jaxlinop.identity_linear_operator import IdentityLinearOperator
from jaxlinop.constant_diagonal_linear_operator import ConstantDiagonalLinearOperator
from jaxlinop.dense_linear_operator import DenseLinearOperator


def approx_equal(res: jax.Array, actual: jax.Array) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-6


@pytest.mark.parametrize("n", [1, 2, 5])
def test_init(n: int) -> None:
    id = IdentityLinearOperator(size=n)
    assert id.shape == (n, n)
    assert id.size == n


@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag(n: int) -> None:
    id = IdentityLinearOperator(size=n)
    res = id.diagonal()
    actual = jnp.ones(n)
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    id = IdentityLinearOperator(size=n)
    actual = jnp.eye(n)
    res = id.to_dense()
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add_diagonal(n: int) -> None:
    id = IdentityLinearOperator(size=n)
    entries = jr.uniform(_PRNGKey, shape=(n,))
    diag = DiagonalLinearOperator(diag=entries)
    id_add_diag = id._add_diagonal(diag)

    assert isinstance(id_add_diag, DiagonalLinearOperator)

    res = id_add_diag.to_dense()
    actual = jnp.eye(n) * (1.0 + entries)
    assert approx_equal(res, actual)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_add(n: int) -> None:

    array = jr.uniform(_PRNGKey, shape=(n, n))
    entries = jr.uniform(_PRNGKey, shape=(n,))
    id = IdentityLinearOperator(size=n)

    # Add array.
    res_left = id + array
    res_right = array + id

    actual = array + jnp.eye(n)

    assert isinstance(res_left, DenseLinearOperator)
    assert isinstance(res_right, DenseLinearOperator)
    assert approx_equal(res_left.to_dense(), actual)
    assert approx_equal(res_right.to_dense(), actual)

    # Add Dense.
    dense_lo = DenseLinearOperator(matrix=array)

    res_left = id + dense_lo
    res_right = dense_lo + id
    actual = DenseLinearOperator(matrix=array + jnp.eye(n))

    assert isinstance(res_left, DenseLinearOperator)
    assert isinstance(res_right, DenseLinearOperator)
    assert approx_equal(res_left.to_dense(), actual.to_dense())
    assert approx_equal(res_right.to_dense(), actual.to_dense())

    # Add Diagonal.
    diag = DiagonalLinearOperator(diag=entries)

    res_left = id + diag
    res_right = diag + id
    actual = DiagonalLinearOperator(diag=entries + jnp.ones(n))

    assert isinstance(res_left, DiagonalLinearOperator)
    assert isinstance(res_right, DiagonalLinearOperator)
    assert approx_equal(res_left.to_dense(), actual.to_dense())
    assert approx_equal(res_right.to_dense(), actual.to_dense())


@pytest.mark.parametrize("n", [1, 2, 5])
def test_mul(n: int) -> None:
    constant = jr.uniform(_PRNGKey, shape=())
    id = IdentityLinearOperator(size=n)

    res_left = id * constant
    res_right = constant * id

    assert isinstance(res_left, ConstantDiagonalLinearOperator)
    assert isinstance(res_right, ConstantDiagonalLinearOperator)
    assert approx_equal(res_left.to_dense(), constant * jnp.eye(n))
    assert approx_equal(res_right.to_dense(), constant * jnp.eye(n))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_matmul(n: int) -> None:
    array = jr.uniform(_PRNGKey, shape=(n, n))
    id = IdentityLinearOperator(size=n)
    res_left = id @ array
    res_right = array @ id
    assert isinstance(res_left, jax.Array)
    assert isinstance(res_right, jax.Array)
    assert approx_equal(res_left, array)
    assert approx_equal(res_right, array)


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_solve(n: int, m: int) -> None:
    id = IdentityLinearOperator(size=n)
    rhs = jr.uniform(_PRNGKey, shape=(n, m))
    res = id.solve(rhs)

    assert isinstance(res, jax.Array)
    assert approx_equal(res, rhs)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_root(n: int) -> None:
    id = IdentityLinearOperator(size=n)
    res = id.to_root()
    assert isinstance(res, IdentityLinearOperator)
    assert approx_equal(res.to_dense(), jnp.eye(n))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_root(n: int) -> None:
    id = IdentityLinearOperator(size=n)
    res = IdentityLinearOperator.from_root(id)
    assert isinstance(res, IdentityLinearOperator)
    assert approx_equal(res.to_dense(), jnp.eye(n))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_from_dense(n: int) -> None:
    dense = jr.uniform(_PRNGKey, shape=(n, n))
    res = IdentityLinearOperator.from_dense(dense)
    assert isinstance(res, IdentityLinearOperator)
    assert approx_equal(res.to_dense(), jnp.eye(n))


__all__ = [
    "IdentityLinearOperator",
]
