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
from beartype.typing import List
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
    AdditiveKernel,
    Constant,
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

@pytest.mark.parametrize("constant", [0.1,10.0])
def test_constant_kernel(constant: Float) -> None:
    k = Constant(constant=jnp.array(constant))
    assert k(jnp.array([1.0]), jnp.array([2.0])) == constant

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

def test_additive_kernel() -> None:

    # test raises if given wrong init
    with pytest.raises(ValueError):
        AdditiveKernel(
            kernels=[RBF()], 
            max_interaction_depth = 2,
            interaction_variances = jnp.array([1.0,1.0])
        )

    base_kernels = [
        RBF(lengthscale=0.1, active_dims=[0]), 
        RBF(lengthscale=1.0, active_dims=[1]), 
        RBF(lengthscale=10.0, active_dims=[2])
        ]
    vars= jnp.array([1.0, 2.0, 3.0, 4.0])
    additive_kernel = AdditiveKernel(
        kernels=base_kernels, 
        max_interaction_depth = 3,
        interaction_variances = vars,
    ) 
    x= jnp.array([1.0,2.0,3.0], dtype=jnp.float64)
    y = jnp.array([3.0,2.0,1.0], dtype=jnp.float64)
    ks = jnp.stack([k(x, y) for k in base_kernels])
    
    # test that the internal Newton Girad identity is correct
    ng_identity = additive_kernel._compute_additive_terms_girad_newton(ks)
    assert ng_identity[0] == 1.0
    all_first = ks[0] + ks[1] + ks[2]
    assert ng_identity[1] == all_first
    all_2nd = ks[0]*ks[1] + ks[0]*ks[2] + ks[1]*ks[2]
    assert (ng_identity[2] - all_2nd)**2 < 1e-10
    all_3rd = ks[0]*ks[1]*ks[2]
    assert (ng_identity[3] - all_3rd)**2 < 1e-10

    # check that the kernel eval is correct
    k_eval = additive_kernel(x, y)
    k_exact = vars[0] + vars[1]*all_first + vars[2]*all_2nd + vars[3]*all_3rd
    assert (k_exact - k_eval)**2 < 1e-10

    # check that get_specific_kernel works
    k0 = additive_kernel.get_specific_kernel([])
    assert isinstance(k0, Constant)
    assert k0(x,y) == vars[0]
    k1 = additive_kernel.get_specific_kernel([0])
    assert isinstance(k1, CombinationKernel)
    assert k1(x,y) == vars[1]* base_kernels[0](x,y)
    k2 = additive_kernel.get_specific_kernel([0,1])
    assert isinstance(k2, CombinationKernel)
    assert k2(x,y) == vars[2]* base_kernels[0](x,y)*base_kernels[1](x,y)





