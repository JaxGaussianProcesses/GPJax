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

from dataclasses import (
    dataclass,
    field,
)

from jax import config
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.base import param_field
from gpjax.kernels.base import (
    AbstractKernel,
    CombinationKernel,
    ProductKernel,
    SumKernel,
)
from gpjax.kernels.nonstationary import (
    Linear,
    Polynomial,
)
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_abstract_kernel():
    # Test initialising abstract kernel raises TypeError with unimplemented __call__ method:
    with pytest.raises(TypeError):
        AbstractKernel()

    # Create a dummy kernel class with __call__ implemented:
    @dataclass
    class DummyKernel(AbstractKernel):
        test_a: Float[Array, "1"] = field(default_factory=lambda: jnp.array([1.0]))
        test_b: Float[Array, "1"] = param_field(
            jnp.array([2.0]), bijector=tfb.Softplus()
        )

        def __call__(
            self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
        ) -> Float[Array, "1"]:
            return x * self.test_b * y

    # Initialise dummy kernel class and test __call__ method:
    dummy_kernel = DummyKernel()
    assert dummy_kernel.test_a == jnp.array([1.0])
    assert isinstance(
        dummy_kernel._pytree__meta["test_b"].get("bijector"), tfb.Softplus
    )
    assert dummy_kernel.test_b == jnp.array([2.0])
    assert dummy_kernel(jnp.array([1.0]), jnp.array([2.0])) == 4.0


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
    kernels = [kernel() for _ in range(n_kerns)]

    # Create combination kernel
    combination_kernel = combination_type(kernels=kernels)

    # Check params are a list of dictionaries
    assert combination_kernel.kernels == kernels

    # Check combination kernel set
    assert len(combination_kernel.kernels) == n_kerns
    assert isinstance(combination_kernel.kernels, list)
    assert isinstance(combination_kernel.kernels[0], AbstractKernel)

    # Compute gram matrix
    Kxx = combination_kernel.gram(x)

    # Check shapes
    assert Kxx.shape[0] == Kxx.shape[1]
    assert Kxx.shape[1] == n

    # Check positive definiteness
    jitter = 1e-6
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense() + jnp.eye(n) * jitter)
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
    sum_kernel = SumKernel(kernels=[k1, k2])

    # Compute gram matrix
    Kxx = sum_kernel.gram(x)

    # Compute gram matrix
    Kxx_k1 = k1.gram(x)
    Kxx_k2 = k2.gram(x)

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
    prod_kernel = ProductKernel(kernels=[k1, k2])

    # Compute gram matrix
    Kxx = prod_kernel.gram(x)

    # Compute gram matrix
    Kxx_k1 = k1.gram(x)
    Kxx_k2 = k2.gram(x)

    # Check manual and automatic gram matrices are equal
    assert jnp.all(Kxx.to_dense() == Kxx_k1.to_dense() * Kxx_k2.to_dense())
