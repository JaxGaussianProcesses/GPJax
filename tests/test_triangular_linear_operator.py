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
from jax.random import KeyArray

# Test settings:
key: KeyArray = jr.PRNGKey(seed=42)
jitter: float = 1e-6
atol: float = 1e-6
config.update("jax_enable_x64", True)

from jaxlinop.triangular_linear_operator import (
    LowerTriangularLinearOperator,
    UpperTriangularLinearOperator,
)
from jaxlinop.dense_linear_operator import DenseLinearOperator
from jaxlinop.diagonal_linear_operator import DiagonalLinearOperator
