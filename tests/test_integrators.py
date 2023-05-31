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


import jax
import jax.random as jr
from jax.config import config
import jax.numpy as jnp
import pytest

from gpjax.integrators import GHQuadratureIntegrator

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("num_points", [10, 20, 30])
def test_quadrature(jit: bool, num_points: int):
    key = jr.PRNGKey(123)

    def test():
        def fun(x, y):
            """In practice, the first argument will be the latent function values"""
            return x**2 + y

        mean = jnp.array([[2.0]])
        variance = jnp.array([[1.0]])
        fn_val = GHQuadratureIntegrator(num_points=num_points).integrate(
            fun=fun,
            mean=mean,
            sigma2=variance,
            y=jnp.ones_like(mean),
        )
        return fn_val.squeeze().round(1)

    if jit:
        test = jax.jit(test)
    assert test() == 6.0


@pytest.mark.parametrize("jit", [True, False])
def test_analytical_gaussian(jit: bool):
    pass
