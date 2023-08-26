from cola.ops import (
    Dense,
    Diagonal,
    I_like,
    Identity,
    Triangular,
)
import jax.numpy as jnp

from gpjax.lower_cholesky import lower_cholesky


def test_dense() -> None:
    array = jnp.array([[3.0, 1.0], [1.0, 3.0]])
    A = Dense(array)

    L = lower_cholesky(A)
    assert isinstance(L, Triangular)
    assert jnp.allclose(L.to_dense(), jnp.linalg.cholesky(array))


def test_diagonal() -> None:
    array = jnp.array([1.0, 2.0])
    A = Diagonal(array)

    L = lower_cholesky(A)
    assert isinstance(L, Diagonal)
    assert jnp.allclose(L.to_dense(), jnp.diag(jnp.sqrt(array)))


def test_identity() -> None:
    A = I_like(jnp.eye(2))
    L = lower_cholesky(A)
    assert isinstance(L, Identity)
    assert jnp.allclose(L.to_dense(), jnp.eye(2))
