# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
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
from jaxlinop import identity

from gpjax.kernels.base import (
    AbstractKernel,
    CombinationKernel,
    ProductKernel,
    SumKernel,
)
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
)
from gpjax.kernels.nonstationary import Polynomial, Linear
from jax.random import KeyArray
from jaxtyping import Array, Float
from typing import Dict


# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.PRNGKey(123)
_jitter = 1e-6


def test_abstract_kernel():
    # Test initialising abstract kernel raises TypeError with unimplemented __call__ and _init_params methods:
    with pytest.raises(TypeError):
        AbstractKernel()

    # Create a dummy kernel class with __call__ and _init_params methods implemented:
    class DummyKernel(AbstractKernel):
        def __call__(
            self, x: Float[Array, "1 D"], y: Float[Array, "1 D"], params: Dict
        ) -> Float[Array, "1"]:
            return x * params["test"] * y

        def init_params(self, key: KeyArray) -> Dict:
            return {"test": 1.0}

    # Initialise dummy kernel class and test __call__ and _init_params methods:
    dummy_kernel = DummyKernel()
    assert dummy_kernel.init_params(_initialise_key) == {"test": 1.0}
    assert (
        dummy_kernel(jnp.array([1.0]), jnp.array([2.0]), {"test": 2.0}) == 4.0
    )


@pytest.mark.parametrize("combination_type", [SumKernel, ProductKernel])
@pytest.mark.parametrize(
    "kernel",
    [RBF, RationalQuadratic, Linear, Matern12, Matern32, Matern52, Polynomial],
)
@pytest.mark.parametrize("n_kerns", [2, 3, 4])
def test_combination_kernel(
    combination_type: CombinationKernel, kernel: AbstractKernel, n_kerns: int
) -> None:

    # Create inputs
    n = 20
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)

    # Create list of kernels
    kernel_set = [kernel() for _ in range(n_kerns)]

    # Create combination kernel
    combination_kernel = combination_type(kernel_set=kernel_set)

    # Initialise default parameters
    params = combination_kernel.init_params(_initialise_key)

    # Check params are a list of dictionaries
    assert len(params) == n_kerns

    for p in params:
        assert isinstance(p, dict)

    # Check combination kernel set
    assert len(combination_kernel.kernel_set) == n_kerns
    assert isinstance(combination_kernel.kernel_set, list)
    assert isinstance(combination_kernel.kernel_set[0], AbstractKernel)

    # Compute gram matrix
    Kxx = combination_kernel.gram(params, x)

    # Check shapes
    assert Kxx.shape[0] == Kxx.shape[1]
    assert Kxx.shape[1] == n

    # Check positive definiteness
    Kxx += identity(n) * _jitter
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert (eigen_values > 0).all()


@pytest.mark.parametrize(
    "k1", [RBF(), Matern12(), Matern32(), Matern52(), Polynomial()]
)
@pytest.mark.parametrize(
    "k2", [RBF(), Matern12(), Matern32(), Matern52(), Polynomial()]
)
def test_sum_kern_value(k1: AbstractKernel, k2: AbstractKernel) -> None:
    # Create inputs
    n = 10
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)

    # Create sum kernel
    sum_kernel = SumKernel(kernel_set=[k1, k2])

    # Initialise default parameters
    params = sum_kernel.init_params(_initialise_key)

    # Compute gram matrix
    Kxx = sum_kernel.gram(params, x)

    # NOW we do the same thing manually and check they are equal:
    # Initialise default parameters
    k1_params = k1.init_params(_initialise_key)
    k2_params = k2.init_params(_initialise_key)

    # Compute gram matrix
    Kxx_k1 = k1.gram(k1_params, x)
    Kxx_k2 = k2.gram(k2_params, x)

    # Check manual and automatic gram matrices are equal
    assert jnp.all(Kxx.to_dense() == Kxx_k1.to_dense() + Kxx_k2.to_dense())


@pytest.mark.parametrize(
    "k1",
    [
        RBF(),
        Matern12(),
        Matern32(),
        Matern52(),
        Polynomial(),
        Linear(),
        Polynomial(),
        RationalQuadratic(),
    ],
)
@pytest.mark.parametrize(
    "k2",
    [
        RBF(),
        Matern12(),
        Matern32(),
        Matern52(),
        Polynomial(),
        Linear(),
        Polynomial(),
        RationalQuadratic(),
    ],
)
def test_prod_kern_value(k1: AbstractKernel, k2: AbstractKernel) -> None:

    # Create inputs
    n = 10
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)

    # Create product kernel
    prod_kernel = ProductKernel(kernel_set=[k1, k2])

    # Initialise default parameters
    params = prod_kernel.init_params(_initialise_key)

    # Compute gram matrix
    Kxx = prod_kernel.gram(params, x)

    # NOW we do the same thing manually and check they are equal:

    # Initialise default parameters
    k1_params = k1.init_params(_initialise_key)
    k2_params = k2.init_params(_initialise_key)

    # Compute gram matrix
    Kxx_k1 = k1.gram(k1_params, x)
    Kxx_k2 = k2.gram(k2_params, x)

    # Check manual and automatic gram matrices are equal
    assert jnp.all(Kxx.to_dense() == Kxx_k1.to_dense() * Kxx_k2.to_dense())


@pytest.mark.parametrize(
    "kernel",
    [RBF, Matern12, Matern32, Matern52, Polynomial, Linear, RationalQuadratic],
)
def test_combination_kernel_type(kernel: AbstractKernel) -> None:
    prod_kern = kernel() * kernel()
    assert isinstance(prod_kern, ProductKernel)
    assert isinstance(prod_kern, CombinationKernel)

    add_kern = kernel() + kernel()
    assert isinstance(add_kern, SumKernel)
    assert isinstance(add_kern, CombinationKernel)
