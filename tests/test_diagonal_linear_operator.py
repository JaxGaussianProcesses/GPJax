# Copyright 2022 The Jax Linear Operator Contributors. All Rights Reserved.
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
from jax_linear_operator.diagonal_linear_operator import DiagonalLinearOperator, I
from jax_linear_operator.dense_linear_operator import DenseLinearOperator

_key = jr.PRNGKey(seed=42)


@pytest.mark.parametrize("n", [1, 10, 100])
def test_adding_jax_arrays(n: int) -> None:
    import jax.random as jr

    # Create PSD jax arrays matricies A and B:
    key_a, key_b = jr.split(_key)

    diag_A = jr.uniform(key_a, (n,))
    diag_B = jr.uniform(key_b, (n,))

    A_mat = jnp.diag(diag_A)
    B_mat = jnp.diag(diag_B)

    # Create corresponding linear operators:
    A_LO = DiagonalLinearOperator(diag=diag_A)
    B_LO = DiagonalLinearOperator(diag=diag_B)

    # Test addition:
    assert jnp.all((A_LO + B_mat).to_dense() == A_mat + B_mat)
    assert jnp.all((B_mat + A_LO).to_dense() == B_mat + A_mat)
    assert jnp.all((A_LO + B_LO).to_dense() == A_mat + B_mat)

    # Test subtraction:
    assert jnp.all((A_LO - B_LO).to_dense() == A_mat - B_mat)
    assert jnp.all((A_LO - B_mat).to_dense() == A_mat - B_mat)
    assert jnp.all((B_mat - A_LO).to_dense() == B_mat - A_mat)


@pytest.mark.parametrize("constant", [1.0, 3.5])
@pytest.mark.parametrize("n", [1, 10, 100])
def test_diagonal_covariance_operator(n: int, constant: float) -> None:
    diag = 1.0 + jnp.arange(n, dtype=jnp.float64)
    diag_cov = DiagonalLinearOperator(diag=diag)

    # Test shape:
    assert diag_cov.shape == (n, n)

    # Test trace:
    assert jnp.allclose(jnp.sum(diag), diag_cov.trace())

    # Test diagonal:
    assert jnp.allclose(diag, diag_cov.diagonal())

    # Test multiplying with scalar:
    assert ((diag_cov * constant).diagonal() == constant * diag).all()

    # Test solve:
    assert (jnp.diagonal(diag_cov.solve(rhs=jnp.eye(n))) == 1.0 / diag).all()

    # Test to_dense method:
    dense = diag_cov.to_dense()
    assert (dense - jnp.diag(diag) == 0.0).all()
    assert dense.shape == (n, n)

    # Test log determinant:
    assert diag_cov.log_det() == 2.0 * jnp.sum(jnp.log(diag))

    # Test lower triangular:
    L = diag_cov.triangular_lower()
    assert L.shape == (n, n)
    assert (L == jnp.diag(jnp.sqrt(diag))).all()

    # Test adding two diagonal covariance operators:
    diag_other = 5.1 + 2 * jnp.arange(n, dtype=jnp.float64)
    other = DiagonalLinearOperator(diag=diag_other)
    assert ((diag_cov + other).diagonal() == diag + diag_other).all()


@pytest.mark.parametrize("n", [1, 10, 100])
def test_identity_covariance_operator(n: int) -> None:

    # Create identity matrix of size nxn:
    Identity = I(n)

    # Check iniation:
    assert Identity.diag.shape == (n,)
    assert (Identity.diag == 1.0).all()
    assert isinstance(Identity.diag, jnp.ndarray)
    assert isinstance(Identity, DiagonalLinearOperator)

    # Check iid covariance construction:
    noise = jnp.array([jnp.pi])
    cov = Identity * noise
    assert cov.diag.shape == (n,)
    assert (cov.diag == jnp.pi).all()
    assert isinstance(cov.diag, jnp.ndarray)
    assert isinstance(cov, DiagonalLinearOperator)

    # Check addition to diagonal covariance:
    diag = jnp.arange(n)
    diag_gram_matrix = DiagonalLinearOperator(diag=diag)
    cov = diag_gram_matrix + Identity
    assert cov.diag.shape == (n,)
    assert (cov.diag == (1.0 + jnp.arange(n))).all()
    assert isinstance(cov.diag, jnp.ndarray)
    assert isinstance(cov, DiagonalLinearOperator)

    # Check addition to dense covariance:
    dense = jnp.arange(n**2, dtype=jnp.float64).reshape(n, n)
    dense_matrix = DenseLinearOperator(matrix=dense)
    cov = dense_matrix + (Identity * noise)
    assert cov.matrix.shape == (n, n)
    assert (jnp.diag(cov.matrix) == jnp.diag((noise + dense))).all()
    assert isinstance(cov.matrix, jnp.ndarray)
    assert isinstance(cov, DenseLinearOperator)
