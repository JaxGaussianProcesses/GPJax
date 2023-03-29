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


from itertools import permutations, product

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
import distrax as dx
from jax.config import config
from gpjax.linops import LinearOperator, identity

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    White,
    Periodic,
    PoweredExponential,
    RationalQuadratic,
)
from gpjax.kernels.computations import DenseKernelComputation, DiagonalKernelComputation
from gpjax.kernels.stationary.utils import build_student_t_distribution
from gpjax.parameters.bijectors import Identity, Softplus

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.PRNGKey(123)
_jitter = 1e-6


class BaseTestKernel:
    """A base class that contains all tests applied on stationary kernels."""

    kernel: AbstractKernel
    default_compute_engine = type
    spectral_density_name: str

    def pytest_generate_tests(self, metafunc):
        """This is called automatically by pytest"""
        id_func = lambda x: "-".join([f"{k}={v}" for k, v in x.items()])
        funcarglist = metafunc.cls.params.get(metafunc.function.__name__, None)

        if funcarglist is None:

            return
        else:
            argnames = sorted(funcarglist[0])
            metafunc.parametrize(
                argnames,
                [[funcargs[name] for name in argnames] for funcargs in funcarglist],
                ids=id_func,
            )

    @pytest.mark.parametrize("dim", [None, 1, 3], ids=lambda x: f"dim={x}")
    def test_initialization(self, fields: dict, dim: int) -> None:

        fields = {k: jnp.array([v]) for k, v in fields.items()}

        # number of dimensions
        if dim is None:
            kernel: AbstractKernel = self.kernel(**fields)
            assert kernel.ndims == 1
        else:
            kernel: AbstractKernel = self.kernel(
                active_dims=[i for i in range(dim)], **fields
            )
            assert kernel.ndims == dim

        # compute engine
        assert kernel.compute_engine == self.default_compute_engine

        # properties
        for field, value in fields.items():
            assert getattr(kernel, field) == value

        # pytree
        leaves = jtu.tree_leaves(kernel)
        assert len(leaves) == len(fields)

        # meta
        meta_leaves = kernel._pytree__meta
        assert meta_leaves.keys() == fields.keys()
        for field in fields:
            if field in ["variance", "lengthscale", "period", "alpha"]:
                assert meta_leaves[field]["bijector"] == Softplus
            if field in ["power"]:
                assert meta_leaves[field]["bijector"] == Identity
            assert meta_leaves[field]["trainable"] == True

        # call
        x = jnp.linspace(0.0, 1.0, 10 * kernel.ndims).reshape(10, kernel.ndims)
        kernel(x, x)

    @pytest.mark.parametrize("n", [1, 5], ids=lambda x: f"n={x}")
    @pytest.mark.parametrize("dim", [1, 3], ids=lambda x: f"dim={x}")
    def test_gram(self, dim: int, n: int) -> None:
        kernel: AbstractKernel = self.kernel()
        kernel.gram
        x = jnp.linspace(0.0, 1.0, n * dim).reshape(n, dim)
        Kxx = kernel.gram(x)
        assert isinstance(Kxx, LinearOperator)
        assert Kxx.shape == (n, n)
        assert jnp.all(jnp.linalg.eigvalsh(Kxx.to_dense() + jnp.eye(n) * 1e-6) > 0.0)

    @pytest.mark.parametrize("n_a", [1, 2, 5], ids=lambda x: f"n_a={x}")
    @pytest.mark.parametrize("n_b", [1, 2, 5], ids=lambda x: f"n_b={x}")
    @pytest.mark.parametrize("dim", [1, 2, 5], ids=lambda x: f"dim={x}")
    def test_cross_covariance(self, n_a: int, n_b: int, dim: int) -> None:

        kernel: AbstractKernel = self.kernel()
        a = jnp.linspace(-1.0, 1.0, n_a * dim).reshape(n_a, dim)
        b = jnp.linspace(3.0, 4.0, n_b * dim).reshape(n_b, dim)
        Kab = kernel.cross_covariance(a, b)
        assert isinstance(Kab, jnp.ndarray)
        assert Kab.shape == (n_a, n_b)

    def test_spectral_density(self):

        kernel: AbstractKernel = self.kernel()

        if self.kernel not in [RBF, Matern12, Matern32, Matern52]:
            with pytest.raises(AttributeError):
                kernel.spectral_density
        else:
            sdensity = kernel.spectral_density
            assert sdensity.name == self.spectral_density_name
            assert sdensity.loc == jnp.array(0.0)
            assert sdensity.scale == jnp.array(1.0)


prod = lambda inp: [
    {"fields": dict(zip(inp.keys(), values))} for values in product(*inp.values())
]


class TestRBF(BaseTestKernel):
    kernel = RBF
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation
    spectral_density_name = "Normal"


class TestMatern12(BaseTestKernel):
    kernel = Matern12
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation
    spectral_density_name = "StudentT"


class TestMatern32(BaseTestKernel):
    kernel = Matern32
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation
    spectral_density_name = "StudentT"


class TestMatern52(BaseTestKernel):
    kernel = Matern52
    fields = prod({"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0]})
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation
    spectral_density_name = "StudentT"


class TestWhite(BaseTestKernel):
    kernel = White
    fields = prod({"variance": [0.1, 1.0]})
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation


class TestPeriodic(BaseTestKernel):
    kernel = Periodic
    fields = prod(
        {"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0], "period": [0.1, 1.0]}
    )
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation


class TestPoweredExponential(BaseTestKernel):
    kernel = PoweredExponential
    fields = prod(
        {"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0], "power": [0.1, 2.0]}
    )
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation


class TestRationalQuadratic(BaseTestKernel):
    kernel = RationalQuadratic
    fields = prod(
        {"lengthscale": [0.1, 1.0], "variance": [0.1, 1.0], "alpha": [0.1, 1.0]}
    )
    params = {"test_initialization": fields}
    default_compute_engine = DenseKernelComputation


@pytest.mark.parametrize("smoothness", [1, 2, 3])
def test_build_studentt_dist(smoothness: int) -> None:
    dist = build_student_t_distribution(smoothness)
    assert isinstance(dist, dx.Distribution)
