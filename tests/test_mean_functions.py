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

from typing import Dict

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.config import config
from jaxtyping import Array, Float

from gpjax.mean_functions import AbstractMeanFunction, Constant, Zero
from gpjax.types import PRNGKeyType

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.PRNGKey(123)


def test_abstract_mean_function() -> None:
    # Test that the abstract mean function cannot be instantiated.
    with pytest.raises(TypeError):
        AbstractMeanFunction()

    # Create a dummy mean funcion class with abstract methods implemented.
    class DummyMeanFunction(AbstractMeanFunction):
        def __call__(self, params: Dict, x: Float[Array, "N D"]) ->  Float[Array, "N 1"]:
            return jnp.ones((x.shape[0], 1))
        
        def _initialise_params(self, key: PRNGKeyType) -> Dict:
            return {}

    # Test that the dummy mean function can be instantiated.
    dummy_mean_function = DummyMeanFunction()
    assert isinstance(dummy_mean_function, AbstractMeanFunction)


@pytest.mark.parametrize("mean_function", [Zero, Constant])
@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("n", [1, 2])
def test_shape(mean_function: AbstractMeanFunction, n:int, dim: int) -> None:
    key = _initialise_key

    # Create test inputs.
    x = jnp.linspace(-1.0, 1.0, num=n * dim).reshape(n, dim)

    # Initialise mean function.
    mf = mean_function(output_dim=dim)

    # Initialise parameters.
    params = mf._initialise_params(key)
    assert isinstance(params, dict)

    # Test shape of mean function.
    mu = mf(params, x)
    assert mu.shape[0] == x.shape[0]
    assert mu.shape[1] == dim