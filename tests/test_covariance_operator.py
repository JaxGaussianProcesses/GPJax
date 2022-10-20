# %%
import jax.numpy as jnp
import pytest

# %%
from gpjax.covariance_operator import (
    CovarianceOperator,
    DenseCovarianceOperator,
    DiagonalCovarianceOperator,
    I,
)


# %%
def test_covariance_operator():

    pass


# %%
def test_dense_covariance_operator():

    pass


# %%
def test_diagonal_covariance_operator():

    pass


# %%
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
