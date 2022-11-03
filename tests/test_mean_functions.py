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

import typing as tp

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.config import config

from gpjax.mean_functions import Constant, Zero
from gpjax.parameters import initialise

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


@pytest.mark.parametrize("meanf", [Zero, Constant])
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_shape(meanf, dim):
    key = jr.PRNGKey(123)
    meanf = meanf(output_dim=dim)
    x = jnp.linspace(-1.0, 1.0, num=10).reshape(-1, 1)
    if dim > 1:
        x = jnp.hstack([x] * dim)
    params, _, _ = initialise(meanf, key).unpack()
    mu = meanf(x, params)
    assert mu.shape[0] == x.shape[0]
    assert mu.shape[1] == dim


@pytest.mark.parametrize("meanf", [Zero, Constant])
def test_initialisers(meanf):
    key = jr.PRNGKey(123)
    params, _, _ = initialise(meanf(), key).unpack()
    assert isinstance(params, tp.Dict)
