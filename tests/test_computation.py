import jax.numpy as jnp
import jax.random as jr
import pytest

from jaxkern.computations import (
    DiagonalKernelComputation,
    ConstantDiagonalKernelComputation,
)
from jaxkern.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    PoweredExponential,
    RationalQuadratic,
    Periodic,
)
from jaxkern.nonstationary import Linear, Polynomial


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        Matern12(),
        Matern32(),
        Matern52(),
        RationalQuadratic(),
        PoweredExponential(),
        Periodic(),
        Linear(),
        Polynomial(),
    ],
)
def test_change_computation(kernel):
    x = jnp.linspace(-3.0, 3.0, 5).reshape(-1, 1)
    key = jr.PRNGKey(123)
    params = kernel.init_params(key)

    # The default computation is DenseKernelComputation
    dense_matrix = kernel.gram(params, x).to_dense()
    dense_diagonals = jnp.diag(dense_matrix)

    # Let's now change the computation to DiagonalKernelComputation
    kernel.compute_engine = DiagonalKernelComputation
    diagonal_matrix = kernel.gram(params, x).to_dense()
    diag_entries = jnp.diag(diagonal_matrix)

    # The diagonal entries should be the same as the dense matrix
    assert jnp.allclose(diag_entries, dense_diagonals)

    # All the off diagonal entries should be zero
    assert jnp.allclose(diagonal_matrix - jnp.diag(diag_entries), 0.0)

    # Let's now change the computation to ConstantDiagonalKernelComputation
    kernel.compute_engine = ConstantDiagonalKernelComputation
    constant_diagonal_matrix = kernel.gram(params, x).to_dense()
    constant_entries = jnp.diag(constant_diagonal_matrix)

    # Assert all the diagonal entries are the same
    assert jnp.allclose(constant_entries, constant_entries[0])

    # All the off diagonal entries should be zero
    assert jnp.allclose(constant_diagonal_matrix - jnp.diag(constant_entries), 0.0)
