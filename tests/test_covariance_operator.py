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


import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax.covariance_operator import (
    CovarianceOperator,
    DenseCovarianceOperator,
    DiagonalCovarianceOperator,
    I,
)


def test_covariance_operator():
    with pytest.raises(TypeError):
        CovarianceOperator()


@pytest.mark.parametrize("n", [1, 10, 100])
def test_dense_covariance_operator(n):

    key = jr.PRNGKey(seed=42)
    A = jr.normal(key, (n, n))
    dense = A.T @ A  # Dense random matrix is positive definite.

    cov = DenseCovarianceOperator(matrix=dense)

    # Test shape:
    assert cov.shape == (n, n)

    # Test solve:
    b = jr.normal(key, (n, 1))
    x = cov.solve(b)
    assert jnp.allclose(b, dense @ x)

    # Test to_dense method:
    assert jnp.allclose(dense, cov.to_dense())

    # Test to_diag method:
    assert jnp.allclose(jnp.diag(dense), cov.diagonal())

    # Test log determinant:
    assert jnp.allclose(jnp.linalg.slogdet(dense)[1], cov.log_det())

    # Test trace:
    assert jnp.allclose(jnp.trace(dense), cov.trace())

    # Test lower triangular:
    assert jnp.allclose(jnp.linalg.cholesky(dense), cov.triangular_lower())

    # Test adding diagonal covariance operator to dense linear operator:
    diag = DiagonalCovarianceOperator(diag=jnp.diag(dense))
    cov = cov + (diag * jnp.pi)
    assert jnp.allclose(dense + jnp.pi * jnp.diag(jnp.diag(dense)), cov.to_dense())


@pytest.mark.parametrize("constant", [1.0, 3.5])
@pytest.mark.parametrize("n", [1, 10, 100])
def test_diagonal_covariance_operator(n, constant):
    diag = 1.0 + jnp.arange(n, dtype=jnp.float64)
    diag_cov = DiagonalCovarianceOperator(diag=diag)

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
    other = DiagonalCovarianceOperator(diag=diag_other)
    assert ((diag_cov + other).diagonal() == diag + diag_other).all()


@pytest.mark.parametrize("n", [1, 10, 100])
def test_identity_covariance_operator(n):

    # Create identity matrix of size nxn:
    Identity = I(n)

    # Check iniation:
    assert Identity.diag.shape == (n,)
    assert (Identity.diag == 1.0).all()
    assert isinstance(Identity.diag, jnp.ndarray)
    assert isinstance(Identity, DiagonalCovarianceOperator)

    # Check iid covariance construction:
    noise = jnp.array([jnp.pi])
    cov = Identity * noise
    assert cov.diag.shape == (n,)
    assert (cov.diag == jnp.pi).all()
    assert isinstance(cov.diag, jnp.ndarray)
    assert isinstance(cov, DiagonalCovarianceOperator)

    # Check addition to diagonal covariance:
    diag = jnp.arange(n)
    diag_gram_matrix = DiagonalCovarianceOperator(diag=diag)
    cov = diag_gram_matrix + Identity
    assert cov.diag.shape == (n,)
    assert (cov.diag == (1.0 + jnp.arange(n))).all()
    assert isinstance(cov.diag, jnp.ndarray)
    assert isinstance(cov, DiagonalCovarianceOperator)

    # Check addition to dense covariance:
    dense = jnp.arange(n**2, dtype=jnp.float64).reshape(n, n)
    dense_matrix = DenseCovarianceOperator(matrix=dense)
    cov = dense_matrix + (Identity * noise)
    assert cov.matrix.shape == (n, n)
    assert (jnp.diag(cov.matrix) == jnp.diag((noise + dense))).all()
    assert isinstance(cov.matrix, jnp.ndarray)
    assert isinstance(cov, DenseCovarianceOperator)
