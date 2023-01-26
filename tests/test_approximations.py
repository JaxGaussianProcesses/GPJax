import pytest
from jaxkern.approximations import RFF
from jaxkern.stationary import (
    Matern12,
    Matern32,
    Matern52,
    RBF,
    RationalQuadratic,
    PoweredExponential,
    Periodic,
)
from jaxkern.nonstationary import Polynomial, Linear
from jaxkern.base import AbstractKernel
import jax.random as jr
from jax.config import config
import jax.numpy as jnp
from jaxlinop import DenseLinearOperator
from typing import Tuple
import jax

config.update("jax_enable_x64", True)
_jitter = 1e-5


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("num_basis_fns", [2, 10, 20])
@pytest.mark.parametrize("n_dims", [1, 2, 5])
def test_frequency_sampler(kernel: AbstractKernel, num_basis_fns: int, n_dims: int):
    key = jr.PRNGKey(123)
    base_kernel = kernel(active_dims=list(range(n_dims)))
    approximate = RFF(base_kernel, num_basis_fns)

    params = approximate.init_params(key)
    assert params["frequencies"].shape == (num_basis_fns, n_dims)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("num_basis_fns", [2, 10, 20])
@pytest.mark.parametrize("n_dims", [1, 2, 5])
@pytest.mark.parametrize("n_data", [50, 100])
def test_gram(kernel: AbstractKernel, num_basis_fns: int, n_dims: int, n_data: int):
    key = jr.PRNGKey(123)
    x = jr.uniform(key, shape=(n_data, 1), minval=-3.0, maxval=3.0).reshape(-1, 1)
    if n_dims > 1:
        x = jnp.hstack([x] * n_dims)
    base_kernel = kernel(active_dims=list(range(n_dims)))
    approximate = RFF(base_kernel, num_basis_fns)

    params = approximate.init_params(key)

    linop = approximate.gram(params, x)

    # Check the return type
    assert isinstance(linop, DenseLinearOperator)

    Kxx = linop.to_dense() + jnp.eye(n_data) * _jitter

    # Check that the shape is correct
    assert Kxx.shape == (n_data, n_data)

    # Check that the Gram matrix is PSD
    evals, _ = jnp.linalg.eigh(Kxx)
    assert jnp.all(evals > 0)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("num_basis_fns", [2, 10, 20])
@pytest.mark.parametrize("n_dims", [1, 2, 5])
@pytest.mark.parametrize("n_datas", [(50, 100), (100, 50)])
def test_cross_covariance(
    kernel: AbstractKernel,
    num_basis_fns: int,
    n_dims: int,
    n_datas: Tuple[int, int],
):
    nd1, nd2 = n_datas
    key = jr.PRNGKey(123)
    x1 = jr.uniform(key, shape=(nd1, 1), minval=-3.0, maxval=3.0)
    if n_dims > 1:
        x1 = jnp.hstack([x1] * n_dims)
    x2 = jr.uniform(key, shape=(nd2, 1), minval=-3.0, maxval=3.0)
    if n_dims > 1:
        x2 = jnp.hstack([x2] * n_dims)

    base_kernel = kernel(active_dims=list(range(n_dims)))
    approximate = RFF(base_kernel, num_basis_fns)

    params = approximate.init_params(key)

    Kxx = approximate.cross_covariance(params, x1, x2)

    # Check the return type
    assert isinstance(Kxx, jax.Array)

    # Check that the shape is correct
    assert Kxx.shape == (nd1, nd2)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("n_dim", [1, 2, 5])
def test_improvement(kernel, n_dim):
    n_data = 100
    key = jr.PRNGKey(123)

    x = jr.uniform(key, minval=-3.0, maxval=3.0, shape=(n_data, n_dim))
    base_kernel = kernel(active_dims=list(range(n_dim)))
    exact_params = base_kernel.init_params(key)
    exact_linop = base_kernel.gram(exact_params, x).to_dense()

    crude_approximation = RFF(base_kernel, num_basis_fns=10)
    c_params = crude_approximation.init_params(key)
    c_linop = crude_approximation.gram(c_params, x).to_dense()

    better_approximation = RFF(base_kernel, num_basis_fns=50)
    b_params = better_approximation.init_params(key)
    b_linop = better_approximation.gram(b_params, x).to_dense()

    c_delta = jnp.linalg.norm(exact_linop - c_linop, ord="fro")
    b_delta = jnp.linalg.norm(exact_linop - b_linop, ord="fro")

    # The frobenius norm of the difference between the exact and approximate
    # should improve as we increase the number of basis functions
    assert c_delta > b_delta


@pytest.mark.parametrize("kernel", [RBF(), Matern12(), Matern32(), Matern52()])
def test_exactness(kernel):
    n_data = 100
    key = jr.PRNGKey(123)

    x = jr.uniform(key, minval=-3.0, maxval=3.0, shape=(n_data, 1))
    exact_params = kernel.init_params(key)
    exact_linop = kernel.gram(exact_params, x).to_dense()

    better_approximation = RFF(kernel, num_basis_fns=500)
    b_params = better_approximation.init_params(key)
    b_linop = better_approximation.gram(b_params, x).to_dense()

    max_delta = jnp.max(exact_linop - b_linop)
    assert max_delta < 0.1


@pytest.mark.parametrize(
    "kernel",
    [RationalQuadratic, PoweredExponential, Polynomial, Linear, Periodic],
)
def test_value_error(kernel):
    with pytest.raises(ValueError):
        RFF(kernel(), num_basis_fns=10)
