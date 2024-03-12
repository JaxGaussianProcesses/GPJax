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

from dataclasses import is_dataclass
from itertools import product
from typing import List

from cola.ops import LinearOperator
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.nonstationary import (
    ArcCosine,
    Linear,
    Polynomial,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.key(123)
_jitter = 1e-6


class BaseTestKernel:
    """A base class that contains all tests applied on non-stationary kernels."""

    kernel: AbstractKernel
    default_compute_engine: AbstractKernelComputation
    static_fields: List[str]

    def pytest_generate_tests(self, metafunc):
        """This is called automatically by pytest."""

        # function for pretty test name
        def id_func(x):
            return "-".join([f"{k}={v}" for k, v in x.items()])

        # get arguments for the test function
        funcarglist = metafunc.cls.params.get(metafunc.function.__name__, None)
        if funcarglist is None:
            return
        else:
            # equivalent of pytest.mark.parametrize applied on the metafunction
            metafunc.parametrize("fields", funcarglist, ids=id_func)

    @pytest.mark.parametrize("dim", [None, 1, 3], ids=lambda x: f"dim={x}")
    def test_initialization(self, fields: dict, dim: int) -> None:
        # Check that kernel is a dataclass
        assert is_dataclass(self.kernel)

        # Input fields as JAX arrays
        fields = {k: jnp.array(v) for k, v in fields.items()}

        # Test number of dimensions
        if dim is None:
            kernel: AbstractKernel = self.kernel(**fields)
            assert kernel.ndims == 1
        else:
            kernel: AbstractKernel = self.kernel(active_dims=list(range(dim)), **fields)
            assert kernel.ndims == dim

        # Check default compute engine
        assert kernel.compute_engine == self.default_compute_engine

        # Check properties
        for field, value in fields.items():
            assert getattr(kernel, field) == value

        # Test that pytree returns param_field objects (and not static_field)
        leaves = jtu.tree_leaves(kernel)
        assert len(leaves) == len(set(fields) - set(self.static_fields))

        # Test dtype of params
        for v in leaves:
            assert v.dtype == jnp.float64

        # Check meta leaves
        meta = kernel._pytree__meta
        assert not any(f in meta for f in self.static_fields)
        assert sorted(list(meta.keys())) == sorted(
            set(fields) - set(self.static_fields)
        )

        for field in meta:
            # Bijectors
            if field in ["variance", "shift"]:
                assert isinstance(meta[field]["bijector"], tfb.Softplus)

            # Trainability state
            assert meta[field]["trainable"] is True

        # Test kernel call
        x = jnp.linspace(0.0, 1.0, 10 * kernel.ndims).reshape(10, kernel.ndims)
        jax.vmap(kernel)(x, x)

    @pytest.mark.parametrize("n", [1, 2, 5], ids=lambda x: f"n={x}")
    @pytest.mark.parametrize("dim", [1, 3], ids=lambda x: f"dim={x}")
    def test_gram(self, dim: int, n: int) -> None:
        # Initialise kernel
        kernel: AbstractKernel = self.kernel()

        # Inputs
        x = jnp.linspace(0.0, 1.0, n * dim).reshape(n, dim)

        # Test gram matrix
        Kxx = kernel.gram(x)
        assert isinstance(Kxx, LinearOperator)
        assert Kxx.shape == (n, n)
        assert jnp.all(jnp.linalg.eigvalsh(Kxx.to_dense() + jnp.eye(n) * 1e-6) > 0.0)

    @pytest.mark.parametrize("n_a", [1, 2, 5], ids=lambda x: f"n_a={x}")
    @pytest.mark.parametrize("n_b", [1, 2, 5], ids=lambda x: f"n_b={x}")
    @pytest.mark.parametrize("dim", [1, 2, 5], ids=lambda x: f"dim={x}")
    def test_cross_covariance(self, n_a: int, n_b: int, dim: int) -> None:
        # Initialise kernel
        kernel: AbstractKernel = self.kernel()

        # Inputs
        a = jnp.linspace(-1.0, 1.0, n_a * dim).reshape(n_a, dim)
        b = jnp.linspace(3.0, 4.0, n_b * dim).reshape(n_b, dim)

        # Test cross-covariance
        Kab = kernel.cross_covariance(a, b)
        assert isinstance(Kab, jnp.ndarray)
        assert Kab.shape == (n_a, n_b)


def prod(inp):
    return [
        dict(zip(inp.keys(), values, strict=True)) for values in product(*inp.values())
    ]


class TestLinear(BaseTestKernel):
    kernel = Linear
    fields = prod({"variance": [0.1, 1.0, 2.0]})
    params = {"test_initialization": fields}
    static_fields = []
    default_compute_engine = DenseKernelComputation()


class TestPolynomial(BaseTestKernel):
    kernel = Polynomial
    fields = prod(
        {"variance": [0.1, 1.0, 2.0], "degree": [1, 2, 3], "shift": [1e-6, 0.1, 1.0]}
    )
    static_fields = ["degree"]
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation()


class TestArcCosine(BaseTestKernel):
    kernel = ArcCosine
    fields = prod(
        {
            "variance": [0.1, 1.0],
            "order": [0, 1, 2],
            "weight_variance": [0.1, 1.0],
            "bias_variance": [0.1, 1.0],
        }
    )
    static_fields = ["order"]
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation()

    @pytest.mark.parametrize("order", [-1, 3], ids=lambda x: f"order={x}")
    def test_defaults(self, order: int) -> None:
        with pytest.raises(ValueError):
            self.kernel(order=order)

    @pytest.mark.parametrize("order", [0, 1, 2], ids=lambda x: f"order={x}")
    def test_values_by_monte_carlo_in_special_case(self, order: int) -> None:
        """For certain values of weight variance (1.0) and bias variance (0.0), we can test
        our calculations using the Monte Carlo expansion of the arccosine kernel, e.g.
        see Eq. (1) of https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf.
        """
        kernel: AbstractKernel = self.kernel(
            weight_variance=jnp.array([1.0, 1.0]), bias_variance=1e-25, order=order
        )
        key = jr.key(123)

        # Inputs close(ish) together
        a = jnp.array([[0.0, 0.0]])
        b = jnp.array([[2.0, 2.0]])

        # calc cross-covariance exactly
        Kab_exact = kernel.cross_covariance(a, b)

        # calc cross-covariance using samples
        weights = jax.random.normal(key, (10_000, 2))  # [S, d]
        weights_a = jnp.matmul(weights, a.T)  # [S, 1]
        weights_b = jnp.matmul(weights, b.T)  # [S, 1]
        H_a = jnp.heaviside(weights_a, 0.5)
        H_b = jnp.heaviside(weights_b, 0.5)
        integrands = H_a * H_b * (weights_a**order) * (weights_b**order)
        Kab_approx = 2.0 * jnp.mean(integrands)

        assert jnp.max(Kab_approx - Kab_exact) < 1e-4
