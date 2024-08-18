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

from itertools import product
from typing import Any

from cola.ops.operator_base import LinearOperator
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.kernels.nonstationary import (
    ArcCosine,
    Linear,
    Polynomial,
)
from gpjax.parameters import (
    PositiveReal,
    Static,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def params_product(params: dict[str, list]) -> list[dict[str, Any]]:
    return [
        dict(zip(params.keys(), values, strict=False))
        for values in product(*params.values())
    ]


TESTED_KERNELS = [
    (
        ArcCosine,
        params_product(
            {
                "order": [0, 1, 2],
                "weight_variance": [0.1, 1.0],
                "bias_variance": [0.1, 1.0],
            }
        ),
    ),
    (Linear, [{}]),
    (Polynomial, params_product({"degree": [1, 2, 3], "shift": [1e-6, 0.1, 1.0]})),
]

VARIANCES = [0.1]


@pytest.fixture
def kernel_request(
    kernel,
    params,
    variance,
):
    return kernel, params, variance


@pytest.mark.parametrize(
    "kernel, params", [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("variance", VARIANCES)
@pytest.fixture
def test_init(kernel_request):
    kernel, params, variance = kernel_request
    return kernel(**params, variance=variance)


@pytest.mark.parametrize(
    "kernel, params", [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("variance", VARIANCES)
def test_init_override_paramtype(kernel_request):
    kernel, params, variance = kernel_request

    new_params = {}  # otherwise we change the fixture and next test fails
    for param, value in params.items():
        if param in ("degree", "order"):
            continue
        new_params[param] = Static(value)

    k = kernel(**new_params, variance=PositiveReal(variance))
    assert isinstance(k.variance, PositiveReal)

    for param in params.keys():
        if param in ("degree", "order"):
            continue
        assert isinstance(getattr(k, param), Static)


@pytest.mark.parametrize("kernel", [k[0] for k in TESTED_KERNELS])
def test_init_defaults(kernel: type[AbstractKernel]):
    # Initialise kernel
    k = kernel()

    # Check that the parameters are set correctly
    assert isinstance(k.compute_engine, type(AbstractKernelComputation()))
    assert isinstance(k.variance, PositiveReal)


@pytest.mark.parametrize("kernel", [k[0] for k in TESTED_KERNELS])
@pytest.mark.parametrize("variance", VARIANCES)
def test_init_variances(kernel: type[AbstractKernel], variance):
    # Initialise kernel
    k = kernel(variance=variance)

    # Check that the parameters are set correctly
    assert isinstance(k.variance, PositiveReal)
    assert jnp.allclose(k.variance.value, jnp.asarray(variance))

    # Check that error is raised if variance is not valid
    with pytest.raises(ValueError):
        k = kernel(variance=-1.0)

    with pytest.raises(TypeError):
        k = kernel(variance=jnp.ones((2, 2)))

    with pytest.raises(TypeError):
        k = kernel(variance="invalid type")


@pytest.mark.parametrize(
    "kernel, params", [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("variance", VARIANCES)
@pytest.mark.parametrize("n", [1, 2, 5], ids=lambda x: f"n={x}")
def test_gram(test_init: AbstractKernel, n: int):
    # kernel is initialized in the test_init fixture
    k = test_init
    n_dims = k.n_dims or 1

    # Inputs
    x = jnp.linspace(0.0, 1.0, n * n_dims).reshape(n, n_dims)

    # Test gram matrix
    Kxx = k.gram(x)
    assert isinstance(Kxx, LinearOperator)
    assert Kxx.shape == (n, n)
    assert jnp.all(jnp.linalg.eigvalsh(Kxx.to_dense() + jnp.eye(n) * 1e-6) > 0.0)


@pytest.mark.parametrize(
    "kernel, params", [(cls, p) for cls, params in TESTED_KERNELS for p in params]
)
@pytest.mark.parametrize("variance", VARIANCES)
@pytest.mark.parametrize("n_a", [1, 2, 5], ids=lambda x: f"n_a={x}")
@pytest.mark.parametrize("n_b", [1, 2, 5], ids=lambda x: f"n_b={x}")
def test_cross_covariance(test_init: AbstractKernel, n_a: int, n_b: int):
    # kernel is initialized in the test_init fixture
    k = test_init
    n_dims = k.n_dims or 1

    # Inputs
    x = jnp.linspace(0.0, 1.0, n_a * n_dims).reshape(n_a, n_dims)
    y = jnp.linspace(0.0, 1.0, n_b * n_dims).reshape(n_b, n_dims)

    # Test cross covariance matrix
    Kxy = k.cross_covariance(x, y)
    assert isinstance(Kxy, jax.Array)
    assert Kxy.shape == (n_a, n_b)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_arccosine_special_case(order: int):
    """For certain values of weight variance (1.0) and bias variance (0.0), we can test
    our calculations using the Monte Carlo expansion of the arccosine kernel, e.g.
    see Eq. (1) of https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf.
    """
    kernel = ArcCosine(
        weight_variance=jnp.array([1.0, 1.0]), bias_variance=1e-25, order=order
    )

    # Inputs close(ish) together
    a = jnp.array([[0.0, 0.0]])
    b = jnp.array([[2.0, 2.0]])

    # calc cross-covariance exactly
    Kab_exact = kernel.cross_covariance(a, b)

    # calc cross-covariance using samples
    weights = jax.random.normal(jr.PRNGKey(123), (10_000, 2))  # [S, d]
    weights_a = jnp.matmul(weights, a.T)  # [S, 1]
    weights_b = jnp.matmul(weights, b.T)  # [S, 1]
    H_a = jnp.heaviside(weights_a, 0.5)
    H_b = jnp.heaviside(weights_b, 0.5)
    integrands = H_a * H_b * (weights_a**order) * (weights_b**order)
    Kab_approx = 2.0 * jnp.mean(integrands)

    assert jnp.max(Kab_approx - Kab_exact) < 1e-4
