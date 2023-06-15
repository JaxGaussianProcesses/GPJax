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

from jax.config import config
import jax.numpy as jnp
import jax.random as jr
from jax.random import KeyArray

# Test settings:
key: KeyArray = jr.PRNGKey(seed=42)
jitter: float = 1e-6
atol: float = 1e-6
config.update("jax_enable_x64", True)

from gpjax.linops.triangular import (
    LowerTriangular,
    UpperTriangular,
)


def test_triangular():
    A = jr.normal(key, (5, 5))
    B = A @ A.T
    sqrt = jnp.linalg.cholesky(B)

    L = LowerTriangular(sqrt)
    U = UpperTriangular(sqrt.T)

    assert jnp.allclose((L @ U).to_dense(), B, atol=atol)
    assert isinstance(L.T, UpperTriangular)
    assert isinstance(U.T, LowerTriangular)
    assert jnp.allclose(L.T.to_dense(), U.to_dense(), atol=atol)
    assert jnp.allclose(U.T.to_dense(), L.to_dense(), atol=atol)
