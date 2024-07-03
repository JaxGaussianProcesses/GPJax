# Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
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
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from gpjax.decision_making.utils import (
    OBJECTIVE,
    build_function_evaluator,
)
from gpjax.typing import (
    Array,
    Float,
)


def test_build_function_evaluator():
    def _square(x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        return x**2

    def _cube(x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        return x**3

    functions = {OBJECTIVE: _square, "CONSTRAINT": _cube}
    fn_evaluator = build_function_evaluator(functions)
    x = jnp.array([[2.0, 3.0]])
    datasets = fn_evaluator(x)
    assert datasets.keys() == functions.keys()
    assert jnp.equal(datasets[OBJECTIVE].X, x).all()
    assert jnp.equal(datasets[OBJECTIVE].y, _square(x)).all()
    assert jnp.equal(datasets["CONSTRAINT"].X, x).all()
    assert jnp.equal(datasets["CONSTRAINT"].y, _cube(x)).all()
