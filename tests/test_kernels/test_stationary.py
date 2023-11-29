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


from dataclasses import is_dataclass
from itertools import product

from cola.ops import LinearOperator
import jax
from jax import config
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.kernels.base import AbstractKernel
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
from gpjax.kernels.stationary.utils import build_student_t_distribution

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


class BaseTestKernel:
    """A base class that contains all tests applied on stationary kernels."""

    kernel: AbstractKernel
    default_compute_engine = AbstractKernelComputation
    spectral_density_name: str

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

        # Check pytree structure
        leaves = jtu.tree_leaves(kernel)
        assert len(leaves) == len(fields)

        # Test dtype of params
        for v in leaves:
            assert v.dtype == jnp.float64

        # meta
        meta = kernel._pytree__meta
        assert meta.keys() == fields.keys()
        for field in fields:
            # Bijectors
            if field in ["variance", "lengthscale", "period", "alpha"]:
                assert isinstance(meta[field]["bijector"], tfb.Softplus)
            if field in ["power"]:
                assert isinstance(meta[field]["bijector"], tfb.Sigmoid)

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

    def test_spectral_density(self):
        # Initialise kernel
        kernel: AbstractKernel = self.kernel()

        if self.kernel not in [RBF, Matern12, Matern32, Matern52]:
            # Check that spectral_density property is None
            assert not kernel.spectral_density
        else:
            # Check that spectral_density property is correct
            sdensity = kernel.spectral_density
            assert sdensity.name == self.spectral_density_name
            assert sdensity.loc == jnp.array(0.0)
            assert sdensity.scale == jnp.array(1.0)

    @pytest.mark.parametrize("dim", [1, 3], ids=lambda x: f"dim={x}")
    def test_isotropic(self, dim: int):
        # Initialise kernel
        kernel: AbstractKernel = self.kernel(active_dims=list(range(dim)))
        if self.kernel not in [White]:
            assert kernel.lengthscale.shape == ()


def prod(inp):
    return [
        dict(zip(inp.keys(), values, strict=True)) for values in product(*inp.values())
    ]


class TestRBF(BaseTestKernel):
    kernel = RBF
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation()
    spectral_density_name = "Normal"


class TestMatern12(BaseTestKernel):
    kernel = Matern12
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation()
    spectral_density_name = "StudentT"


class TestMatern32(BaseTestKernel):
    kernel = Matern32
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation()
    spectral_density_name = "StudentT"


class TestMatern52(BaseTestKernel):
    kernel = Matern52
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation()
    spectral_density_name = "StudentT"


class TestWhite(BaseTestKernel):
    kernel = White
    fields = prod({"variance": [0.1, 1.0]})
    params = {"test_initialization": fields}
    default_compute_engine = ConstantDiagonalKernelComputation()


class TestPeriodic(BaseTestKernel):
    kernel = Periodic
    fields = prod(
        {"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0], "period": [0.1, 1.0]}
    )
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation()


class TestPoweredExponential(BaseTestKernel):
    kernel = PoweredExponential
    fields = prod(
        {"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0], "power": [0.1, 0.9]}
    )
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation()


class TestRationalQuadratic(BaseTestKernel):
    kernel = RationalQuadratic
    fields = prod(
        {"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0], "alpha": [0.1, 1.0]}
    )
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation()


@pytest.mark.parametrize("smoothness", [1, 2, 3])
def test_build_studentt_dist(smoothness: int) -> None:
    dist = build_student_t_distribution(smoothness)
    assert isinstance(dist, tfd.Distribution)
