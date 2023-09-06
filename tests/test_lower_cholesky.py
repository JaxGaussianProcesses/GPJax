import cola
from cola.ops import (
    BlockDiag,
    Dense,
    Diagonal,
    I_like,
    Identity,
    Kronecker,
    Triangular,
)
import jax.numpy as jnp
import pytest

from gpjax.lower_cholesky import lower_cholesky


def test_dense() -> None:
    array = jnp.array([[3.0, 1.0], [1.0, 3.0]])

    # Test that we get an error if we don't use cola.PSD!
    with pytest.raises(ValueError):
        A = Dense(array)
        lower_cholesky(A)

    # Now we annoate with cola.PSD and test for the correct output.
    A = cola.PSD(Dense(array))

    L = lower_cholesky(A)
    assert isinstance(L, Triangular)
    assert jnp.allclose(L.to_dense(), jnp.linalg.cholesky(array))


def test_diagonal() -> None:
    array = jnp.array([1.0, 2.0])
    A = cola.PSD(Diagonal(array))

    L = lower_cholesky(A)
    assert isinstance(L, Diagonal)
    assert jnp.allclose(L.to_dense(), jnp.diag(jnp.sqrt(array)))


def test_identity() -> None:
    A = I_like(jnp.eye(2))
    L = lower_cholesky(A)
    assert isinstance(L, Identity)
    assert jnp.allclose(L.to_dense(), jnp.eye(2))


def test_kronecker() -> None:
    array_a = jnp.array([[3.0, 1.0], [1.0, 3.0]])
    array_b = jnp.array([[2.0, 0.0], [0.0, 2.0]])

    # Create LinearOperators.
    A = Dense(array_a)
    B = Dense(array_b)

    # Annotate with cola.PSD.
    A = cola.PSD(A)
    B = cola.PSD(B)

    K = Kronecker(A, B)

    # Cholesky decomposition.
    L = lower_cholesky(K)

    # Check types.
    assert isinstance(L, Kronecker)
    assert isinstance(L.Ms[0], Triangular)
    assert isinstance(L.Ms[1], Triangular)

    # Check values.
    assert jnp.allclose(L.Ms[0].to_dense(), jnp.linalg.cholesky(array_a))
    assert jnp.allclose(L.Ms[1].to_dense(), jnp.linalg.cholesky(array_b))
    assert jnp.allclose(
        L.to_dense(),
        jnp.kron(jnp.linalg.cholesky(array_a), jnp.linalg.cholesky(array_b)),
    )


def test_block_diag() -> None:
    array_a = jnp.array([[3.0, 1.0], [1.0, 3.0]])
    array_b = jnp.array([[2.0, 0.0], [0.0, 2.0]])

    # Create LinearOperators.
    A = Dense(array_a)
    B = Dense(array_b)

    # Annotate with cola.PSD.
    A = cola.PSD(A)
    B = cola.PSD(B)

    B = BlockDiag(A, B, multiplicities=[2, 3])

    # Cholesky decomposition.
    L = lower_cholesky(B)

    # Check types.
    assert isinstance(L, BlockDiag)
    assert isinstance(L.Ms[0], Triangular)
    assert isinstance(L.Ms[1], Triangular)

    # Check values.
    assert jnp.allclose(L.Ms[0].to_dense(), jnp.linalg.cholesky(array_a))
    assert jnp.allclose(L.Ms[1].to_dense(), jnp.linalg.cholesky(array_b))

    # Check multiplicities.
    assert L.multiplicities == [2, 3]

    # Check dense.
    assert jnp.allclose(jnp.linalg.cholesky(B.to_dense()), L.to_dense())
