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
from jax.config import config
import jax.numpy as jnp
import pytest

from gpjax.quadrature import gauss_hermite_quadrature

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


@pytest.mark.parametrize("jit", [True, False])
def test_quadrature(jit):
    def test():
        def fun(x):
            return x**2

        mean = jnp.array([[2.0]])
        var = jnp.array([[1.0]])
        fn_val = gauss_hermite_quadrature(fun, mean, var)
        return fn_val.squeeze().round(1)

    if jit:
        test = jax.jit(test)
    assert test() == 5.0
