# Copyright 2022 The Jax Linear Operator Contributors. All Rights Reserved.
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


import jax.random as jr
import pytest
from jax.config import config

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
from jax_linear_operator.linear_operator import LinearOperator

_key = jr.PRNGKey(seed=42)


def test_covariance_operator() -> None:
    with pytest.raises(TypeError):
        LinearOperator()
