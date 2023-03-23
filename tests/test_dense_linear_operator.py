# %% [markdown]
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


# %%
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.config import config

# %%
# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

# %%
from jax_linear_operator.dense_linear_operator import DenseLinearOperator
from jax_linear_operator.diagonal_linear_operator import DiagonalLinearOperator

# %%
_key = jr.PRNGKey(seed=42)


# %%
@pytest.mark.parametrize("n", [1, 10, 100])
def test_dense_covariance_operator(n: int) -> None:

    sqrt = jr.normal(_key, (n, n))
    dense = sqrt.T @ sqrt  # Dense random matrix is positive definite.
    cov = DenseLinearOperator(matrix=dense)

    # Test shape:
    assert cov.shape == (n, n)

    # Test solve:
    b = jr.normal(_key, (n, 1))
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
    diag = DiagonalLinearOperator(diag=jnp.diag(dense))
    cov = cov + (diag * jnp.pi)
    assert jnp.allclose(dense + jnp.pi * jnp.diag(jnp.diag(dense)), cov.to_dense())
