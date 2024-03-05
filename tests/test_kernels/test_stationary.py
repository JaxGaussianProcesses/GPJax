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

from cola.ops.operator_base import LinearOperator
import jax
from jax import config
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.kernels.computations import (
    AbstractKernelComputation,
    ConstantDiagonalKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    PoweredExponential,
    RationalQuadratic,
    White,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.stationary.utils import build_student_t_distribution
from gpjax.parameters import (
    Parameter,
    Static,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


class BaseTestKernel:
    """A base class that contains all tests applied on stationary kernels."""

    kernel: StationaryKernel
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
            metafunc.parametrize("fields", funcarglist, ids=id_func)

    @pytest.mark.parametrize(
        "active_dims, n_dims",
        (p := [(1, 1), ([2, 3, 4], 3), (slice(1, 3), 2)]),
        ids=[f"active_dims={x[0]}-n_dims={x[1]}" for x in p],
    )
    def test_init_active_dims(self, active_dims, n_dims) -> None:
        # initialise kernel
        kernel: StationaryKernel = self.kernel(active_dims)

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

    @pytest.mark.parametrize(
        "variance", [-1.0, jnp.array([1.0, 2.0])], ids=lambda x: f"variance={x}"
    )
    def test_init_bad_variance_raises_error(self, variance) -> None:
        # variance must be a scalar
        if len(jnp.shape(variance)) > 0:
            with pytest.raises(TypeError):
                self.kernel(1, variance=variance)
            return

        # variance must be positive
        if variance < 0.0:
            with pytest.raises(ValueError):
                self.kernel(1, variance=variance)

    @pytest.mark.parametrize(
        "lengthscale",
        [-1.0, jnp.array([-1.0, 2.0]), jnp.ones((2, 2))],
        ids=lambda x: f"lengthscale={x}",
    )
    def test_init_bad_lengthscale_raises_error(self, lengthscale) -> None:
        shape = jnp.shape(lengthscale)

        # white kernel does not have lengthscale
        if self.kernel == White:
            with pytest.raises(TypeError):
                self.kernel(1, lengthscale=lengthscale)
            return

        # lengthscale must be a scalar or array with shape (n_dims,)
        if len(shape) > 1:
            with pytest.raises(TypeError):
                self.kernel(1, lengthscale=lengthscale)
            return
        elif len(shape) == 0:
            with pytest.raises(ValueError):
                self.kernel(1, lengthscale=lengthscale)
        else:
            with pytest.raises(ValueError):
                self.kernel(2, lengthscale=lengthscale)

    def test_init_params_defaults(self, fields) -> None:
        # Initialise kernel
        kernel: StationaryKernel = self.kernel(1, **fields)

        # Check that the parameters are set correctly
        for k, v in fields.items():
            param = getattr(kernel, k)
            assert isinstance(param, Parameter)
            assert param.value == jnp.asarray(v)

    def test_init_params_overrides(self, fields) -> None:
        # Initialise kernel
        fields = {k: Static(v) for k, v in fields.items()}
        kernel: StationaryKernel = self.kernel(1, **fields)

        # Check that the parameters are set correctly
        for k, v in fields.items():
            param = getattr(kernel, k)
            assert isinstance(param, Static)
            assert param.value == v.value

    @pytest.mark.parametrize("n", [1, 2, 5], ids=lambda x: f"n={x}")
    @pytest.mark.parametrize(
        "active_dims", [2, [2, 3, 4], slice(1, 3)], ids=lambda x: f"active_dims={x}"
    )
    def test_gram(self, active_dims, n) -> None:
        # Initialise kernel
        kernel: StationaryKernel = self.kernel(active_dims)

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
        kernel: StationaryKernel = self.kernel(active_dims)
        n_dims = kernel.n_dims

        # Inputs
        a = jnp.linspace(-1.0, 1.0, n_a * n_dims).reshape(n_a, n_dims)
        b = jnp.linspace(3.0, 4.0, n_b * n_dims).reshape(n_b, n_dims)

        # Test cross-covariance
        Kab = kernel.cross_covariance(a, b)
        assert isinstance(Kab, jnp.ndarray)
        assert Kab.shape == (n_a, n_b)

    def test_spectral_density(self):
        # Initialise kernel
        kernel: StationaryKernel = self.kernel(1)

        if isinstance(kernel, (RBF, Matern12, Matern32, Matern52)):
            # Check that spectral_density property is correct
            sdensity = kernel.spectral_density
            assert sdensity.name == self.spectral_density_name
            assert sdensity.loc == jnp.array(0.0)
            assert sdensity.scale == jnp.array(1.0)
        elif isinstance(
            kernel, (White, Periodic, PoweredExponential, RationalQuadratic)
        ):
            # Check that spectral_density property is None
            assert kernel.spectral_density is None


def prod(inp):
    return [
        dict(zip(inp.keys(), values, strict=True)) for values in product(*inp.values())
    ]


class TestRBF(BaseTestKernel):
    kernel = RBF
    fields = prod({"lengthscale": [[0.1], 1.0], "variance": [0.1, 1.0]})
    params = {
        "test_init_params_defaults": fields,
        "test_init_params_overrides": fields,
    }
    default_compute_engine = DenseKernelComputation()
    spectral_density_name = "Normal"


class TestMatern12(BaseTestKernel):
    kernel = Matern12
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {
        "test_init_params_defaults": fields,
        "test_init_params_overrides": fields,
    }
    default_compute_engine = DenseKernelComputation()
    spectral_density_name = "StudentT"


class TestMatern32(BaseTestKernel):
    kernel = Matern32
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {
        "test_init_params_defaults": fields,
        "test_init_params_overrides": fields,
    }
    default_compute_engine = DenseKernelComputation()
    spectral_density_name = "StudentT"


class TestMatern52(BaseTestKernel):
    kernel = Matern52
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {
        "test_init_params_defaults": fields,
        "test_init_params_overrides": fields,
    }
    default_compute_engine = DenseKernelComputation()
    spectral_density_name = "StudentT"


class TestWhite(BaseTestKernel):
    kernel = White
    fields = prod({"variance": [0.1, 1.0]})
    params = {
        "test_init_params_defaults": fields,
        "test_init_params_overrides": fields,
    }
    default_compute_engine = ConstantDiagonalKernelComputation()


class TestPeriodic(BaseTestKernel):
    kernel = Periodic
    fields = prod(
        {"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0], "period": [0.1, 1.0]}
    )
    params = {
        "test_init_params_defaults": fields,
        "test_init_params_overrides": fields,
    }
    default_compute_engine = DenseKernelComputation()


class TestPoweredExponential(BaseTestKernel):
    kernel = PoweredExponential
    fields = prod(
        {"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0], "power": [0.1, 0.9]}
    )
    params = {
        "test_init_params_defaults": fields,
        "test_init_params_overrides": fields,
    }
    default_compute_engine = DenseKernelComputation()


class TestRationalQuadratic(BaseTestKernel):
    kernel = RationalQuadratic
    fields = prod(
        {"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0], "alpha": [0.1, 1.0]}
    )
    params = {
        "test_init_params_defaults": fields,
        "test_init_params_overrides": fields,
    }
    default_compute_engine = DenseKernelComputation()


@pytest.mark.parametrize("smoothness", [1, 2, 3])
def test_build_studentt_dist(smoothness: int) -> None:
    dist = build_student_t_distribution(smoothness)
    assert isinstance(dist, tfd.Distribution)
