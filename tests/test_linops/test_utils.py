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

from numbers import Number
from typing import Union

from jax.config import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
)
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_PRNGKey = jr.PRNGKey(42)

from gpjax.linops.dense import Dense
from gpjax.linops.utils import (
    to_dense,
    to_linear_operator,
)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("m", [1, 2])
def test_to_dense(n: int, m: int) -> None:
    array = jr.uniform(_PRNGKey, (n, m))
    lo = Dense(array)

    assert jnp.allclose(to_dense(lo), lo.to_dense())
    assert jnp.allclose(to_dense(array), array)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("m", [1, 2])
def test_array_to_linop(n: int, m: int) -> None:
    array = jr.uniform(_PRNGKey, (n, m))
    lo = to_linear_operator(array)

    assert isinstance(lo, Dense)
    assert jnp.allclose(lo.to_dense(), array)


@pytest.mark.parametrize("number", [1.0, jnp.array(2.0), jnp.array(3.0)])
def test_number_to_linop(
    number: Union[Number, Float[Array, ""], Float[Array, "1"]]
) -> None:
    lo = to_linear_operator(number)
    assert isinstance(lo, Dense)
    assert jnp.allclose(lo.to_dense(), jnp.atleast_1d(number))
