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

from cola.ops.operator_base import LinearOperator
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest

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
from gpjax.parameters import (
    Parameter,
    Static,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.PRNGKey(123)
_jitter = 1e-6


class BaseTestKernel:
    """A base class that contains all tests applied on stationary kernels."""

    kernel: AbstractKernel
    default_compute_engine = AbstractKernelComputation
    spectral_density_name: str

    def pytest_generate_tests(self, metafunc: pytest.Metafunc):
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
            metafunc.parametrize("params", funcarglist, ids=id_func)

    @pytest.mark.parametrize(
        "active_dims, n_dims",
        (p := [(1, 1), ([2, 3, 4], 3), (slice(1, 3), 2)]),
        ids=[f"active_dims={x[0]}-n_dims={x[1]}" for x in p],
    )
    def test_init_active_dims(self, active_dims, n_dims) -> None:
        # initialise kernel
        kernel: AbstractKernel = self.kernel(active_dims)

        # Check n_dims
        assert kernel.n_dims == n_dims

        # Check default compute engine
        assert isinstance(kernel.compute_engine, type(self.default_compute_engine))

        # Test kernel call
        x = jnp.linspace(0.0, 1.0, 10 * kernel.n_dims).reshape(10, kernel.n_dims)
        jax.vmap(kernel)(x, x)

    @pytest.mark.parametrize(
        "active_dims", [None, "missing", slice(None)], ids=lambda x: f"active_dims={x}"
    )
    def test_init_bad_active_dims_raises_error(self, active_dims) -> None:
        # active_dims must be given
        if active_dims == "missing":
            with pytest.raises(TypeError):
                self.kernel()

        # active_dims must be int, list of int, or slice
        if active_dims is None:
            with pytest.raises(TypeError):
                self.kernel(None)

        # active_dims must have "stop" if it is a slice
        if active_dims == slice(None):
            with pytest.raises(ValueError):
                self.kernel(slice(None))

    def test_init_params_defaults(self, params) -> None:
        # Initialise kernel
        kernel: AbstractKernel = self.kernel(1, **params)

        # Check that the parameters are set correctly
        for k, v in params.items():
            if k in self.static_fields:
                continue
            param = getattr(kernel, k)
            assert isinstance(param, Parameter)
            assert param.value == jnp.asarray(v)

    def test_init_params_overrides(self, params) -> None:
        # Initialise kernel
        params = {
            k: Static(v) for k, v in params.items() if k not in self.static_fields
        }
        kernel: AbstractKernel = self.kernel(1, **params)

        # Check that the parameters are set correctly
        for k, v in params.items():
            if k in self.static_fields:
                continue
            param = getattr(kernel, k)
            assert isinstance(param, Static)
            assert param.value == v.value

    @pytest.mark.parametrize("n", [1, 2, 5], ids=lambda x: f"n={x}")
    @pytest.mark.parametrize(
        "active_dims", [2, [2, 3, 4], slice(1, 3)], ids=lambda x: f"active_dims={x}"
    )
    def test_gram(self, active_dims, n) -> None:
        # Initialise kernel
        kernel: AbstractKernel = self.kernel(active_dims)

        # Inputs
        x = jnp.linspace(0.0, 1.0, n * kernel.n_dims).reshape(n, kernel.n_dims)

        # Test gram matrix
        Kxx = kernel.gram(x)
        assert isinstance(Kxx, LinearOperator)
        assert Kxx.shape == (n, n)
        assert jnp.all(jnp.linalg.eigvalsh(Kxx.to_dense() + jnp.eye(n) * 1e-6) > 0.0)

    @pytest.mark.parametrize("n_a", [1, 2, 5], ids=lambda x: f"n_a={x}")
    @pytest.mark.parametrize("n_b", [1, 2, 5], ids=lambda x: f"n_b={x}")
    @pytest.mark.parametrize(
        "active_dims", [2, [2, 3, 4], slice(1, 3)], ids=lambda x: f"active_dims={x}"
    )
    def test_cross_covariance(self, n_a, n_b, active_dims) -> None:
        # Initialise kernel
        kernel: AbstractKernel = self.kernel(active_dims)
        n_dims = kernel.n_dims

        # Inputs
        a = jnp.linspace(-1.0, 1.0, n_a * n_dims).reshape(n_a, n_dims)
        b = jnp.linspace(3.0, 4.0, n_b * n_dims).reshape(n_b, n_dims)

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
    params = {"test_init_params_defaults": fields, "test_init_params_overrides": fields}
    static_fields = []
    default_compute_engine = DenseKernelComputation()


class TestPolynomial(BaseTestKernel):
    kernel = Polynomial
    fields = prod(
        {"variance": [0.1, 1.0, 2.0], "degree": [1, 2, 3], "shift": [1e-6, 0.1, 1.0]}
    )
    static_fields = ["degree"]
    params = {"test_init_params_defaults": fields, "test_init_params_overrides": fields}
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
    params = {"test_init_params_defaults": fields, "test_init_params_overrides": fields}
    default_compute_engine = DenseKernelComputation()

    @pytest.mark.parametrize("order", [-1, 3], ids=lambda x: f"order={x}")
    def test_defaults(self, order: int) -> None:
        with pytest.raises(TypeError):
            self.kernel(1, order=order)

    @pytest.mark.parametrize("order", [0, 1, 2], ids=lambda x: f"order={x}")
    def test_values_by_monte_carlo_in_special_case(self, order: int) -> None:
        """For certain values of weight variance (1.0) and bias variance (0.0), we can test
        our calculations using the Monte Carlo expansion of the arccosine kernel, e.g.
        see Eq. (1) of https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf.
        """
        kernel: AbstractKernel = self.kernel(
            2, weight_variance=jnp.array([1.0, 1.0]), bias_variance=1e-25, order=order
        )
        key = jr.PRNGKey(123)

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
