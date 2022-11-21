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
import pytest
from jax.config import config

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_PRNGKey = jr.PRNGKey(42)

from jaxlinop.identity_linear_operator import IdentityLinearOperator
from jaxlinop.dense_linear_operator import DenseLinearOperator

from jaxlinop.utils import identity, to_dense


@pytest.mark.parametrize("n", [1, 2, 5])
def test_identity(n: int) -> None:
    id = identity(n)
    assert isinstance(id, IdentityLinearOperator)
    assert id.shape == (n, n)
    assert id.size == n


@pytest.mark.parametrize("n", [1, 2, 5])
def test_to_dense(n: int) -> None:
    mat = jr.uniform(_PRNGKey, (n, n))
    lo = DenseLinearOperator(mat)

    assert jnp.allclose(to_dense(lo), lo.to_dense())
    assert jnp.allclose(to_dense(mat), mat)
