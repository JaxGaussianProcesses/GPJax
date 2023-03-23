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


from itertools import permutations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import distrax as dx
from jax.config import config
from gpjax.linops import LinearOperator, identity
from jaxutils.parameters import initialise

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    PoweredExponential,
    RationalQuadratic,
    Periodic,
    White,
)
from gpjax.kernels.stationary.utils import build_student_t_distribution

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.PRNGKey(123)
_jitter = 1e-6


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        Matern12(),
        Matern32(),
        Matern52(),
        RationalQuadratic(),
        White(),
    ],
)
@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("n", [1, 2, 10])
def test_gram(kernel: AbstractKernel, dim: int, n: int) -> None:

    # Gram constructor static method:
    kernel.gram

    # Inputs x:
    x = jnp.linspace(0.0, 1.0, n * dim).reshape(n, dim)

    # Default kernel parameters:
    params = kernel.init_params(_initialise_key)

    # Test gram matrix:
    Kxx = kernel.gram(params, x)
    assert isinstance(Kxx, LinearOperator)
    assert Kxx.shape == (n, n)


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        Matern12(),
        Matern32(),
        Matern52(),
        RationalQuadratic(),
        White(),
    ],
)
@pytest.mark.parametrize("num_a", [1, 2, 5])
@pytest.mark.parametrize("num_b", [1, 2, 5])
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_cross_covariance(
    kernel: AbstractKernel, num_a: int, num_b: int, dim: int
) -> None:
    # Inputs a, b:
    a = jnp.linspace(-1.0, 1.0, num_a * dim).reshape(num_a, dim)
    b = jnp.linspace(3.0, 4.0, num_b * dim).reshape(num_b, dim)

    # Default kernel parameters:
    params = kernel.init_params(_initialise_key)

    # Test cross covariance, Kab:
    Kab = kernel.cross_covariance(params, a, b)
    assert isinstance(Kab, jnp.ndarray)
    assert Kab.shape == (num_a, num_b)


@pytest.mark.parametrize("kernel", [RBF(), Matern12(), Matern32(), Matern52(), White()])
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_call(kernel: AbstractKernel, dim: int) -> None:

    # Datapoint x and datapoint y:
    x = jnp.array([[1.0] * dim])
    y = jnp.array([[0.5] * dim])

    # Defualt parameters:
    params = kernel.init_params(_initialise_key)

    # Test calling gives an autocovariance value of no dimension between the inputs:
    kxy = kernel(params, x, y)

    assert isinstance(kxy, jax.Array)
    assert kxy.shape == ()


@pytest.mark.parametrize("kern", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("ell, sigma", [(0.1, 0.2), (0.5, 0.1), (0.1, 0.5), (0.5, 0.5)])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_pos_def(
    kern: AbstractKernel, dim: int, ell: float, sigma: float, n: int
) -> None:
    kern = kern(active_dims=list(range(dim)))

    # Create inputs x:
    x = jr.uniform(_initialise_key, (n, dim))
    params = {"lengthscale": jnp.array([ell]), "variance": jnp.array([sigma])}

    # Test gram matrix eigenvalues are positive:
    Kxx = kern.gram(params, x)
    Kxx += identity(n) * _jitter
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert (eigen_values > 0.0).all()


@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("ell, sigma", [(0.1, 0.2), (0.5, 0.1), (0.1, 0.5), (0.5, 0.5)])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_pos_def_rq(dim: int, ell: float, sigma: float, alpha: float, n: int) -> None:
    kern = RationalQuadratic(active_dims=list(range(dim)))
    # Gram constructor static method:
    kern.gram

    # Create inputs x:
    x = jr.uniform(_initialise_key, (n, dim))
    params = {
        "lengthscale": jnp.array([ell]),
        "variance": jnp.array([sigma]),
        "alpha": jnp.array([alpha]),
    }

    # Test gram matrix eigenvalues are positive:
    Kxx = kern.gram(params, x)
    Kxx += identity(n) * _jitter
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert (eigen_values > 0.0).all()


@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("ell, sigma", [(0.1, 0.2), (0.5, 0.1), (0.1, 0.5), (0.5, 0.5)])
@pytest.mark.parametrize("period", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_pos_def_periodic(
    dim: int, ell: float, sigma: float, period: float, n: int
) -> None:
    kern = Periodic(active_dims=list(range(dim)))
    # Gram constructor static method:
    kern.gram

    # Create inputs x:
    x = jr.uniform(_initialise_key, (n, dim))
    params = {
        "lengthscale": jnp.array([ell]),
        "variance": jnp.array([sigma]),
        "period": jnp.array([period]),
    }

    # Test gram matrix eigenvalues are positive:
    Kxx = kern.gram(params, x)
    Kxx += identity(n) * _jitter
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert (eigen_values > 0.0).all()


@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("ell, sigma", [(0.1, 0.2), (0.5, 0.1), (0.1, 0.5), (0.5, 0.5)])
@pytest.mark.parametrize("power", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_pos_def_power_exp(
    dim: int, ell: float, sigma: float, power: float, n: int
) -> None:
    kern = PoweredExponential(active_dims=list(range(dim)))
    # Gram constructor static method:
    kern.gram

    # Create inputs x:
    x = jr.uniform(_initialise_key, (n, dim))
    params = {
        "lengthscale": jnp.array([ell]),
        "variance": jnp.array([sigma]),
        "power": jnp.array([power]),
    }

    # Test gram matrix eigenvalues are positive:
    Kxx = kern.gram(params, x)
    Kxx += identity(n) * _jitter
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert (eigen_values > 0.0).all()


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("dim", [None, 1, 2, 5, 10])
def test_initialisation(kernel: AbstractKernel, dim: int) -> None:

    if dim is None:
        kern = kernel()
        assert kern.ndims == 1

    else:
        kern = kernel(active_dims=[i for i in range(dim)])
        params = kern.init_params(_initialise_key)

        assert list(params.keys()) == ["lengthscale", "variance"]
        assert all(params["lengthscale"] == jnp.array([1.0] * dim))
        assert params["variance"] == jnp.array([1.0])

        if dim > 1:
            assert kern.ard
        else:
            assert not kern.ard


@pytest.mark.parametrize(
    "kernel",
    [
        RBF,
        Matern12,
        Matern32,
        Matern52,
        RationalQuadratic,
        Periodic,
        PoweredExponential,
    ],
)
def test_dtype(kernel: AbstractKernel) -> None:
    parameter_state = initialise(kernel(), _initialise_key)
    params, *_ = parameter_state.unpack()
    for k, v in params.items():
        assert v.dtype == jnp.float64
        assert isinstance(k, str)


@pytest.mark.parametrize(
    "kernel",
    [RBF, Matern12, Matern32, Matern52, RationalQuadratic],
)
def test_active_dim(kernel: AbstractKernel) -> None:
    dim_list = [0, 1, 2, 3]
    perm_length = 2
    dim_pairs = list(permutations(dim_list, r=perm_length))
    n_dims = len(dim_list)

    # Generate random inputs
    x = jr.normal(_initialise_key, shape=(20, n_dims))

    for dp in dim_pairs:
        # Take slice of x
        slice = x[..., dp]

        # Define kernels
        ad_kern = kernel(active_dims=dp)
        manual_kern = kernel(active_dims=[i for i in range(perm_length)])

        # Get initial parameters
        ad_params = ad_kern.init_params(_initialise_key)
        manual_params = manual_kern.init_params(_initialise_key)

        # Compute gram matrices
        ad_Kxx = ad_kern.gram(ad_params, x)
        manual_Kxx = manual_kern.gram(manual_params, slice)

        # Test gram matrices are equal
        assert jnp.all(ad_Kxx.to_dense() == manual_Kxx.to_dense())


@pytest.mark.parametrize("smoothness", [1, 2, 3])
def test_build_studentt_dist(smoothness: int) -> None:
    dist = build_student_t_distribution(smoothness)
    assert isinstance(dist, dx.Distribution)


@pytest.mark.parametrize(
    "kern, df", [(Matern12(), 1), (Matern32(), 3), (Matern52(), 5)]
)
def test_matern_spectral_density(kern, df) -> None:
    sdensity = kern.spectral_density
    assert sdensity.name == "StudentT"
    assert sdensity.df == df
    assert sdensity.loc == jnp.array(0.0)
    assert sdensity.scale == jnp.array(1.0)


def test_rbf_spectral_density() -> None:
    kern = RBF()
    sdensity = kern.spectral_density
    assert sdensity.name == "Normal"
    assert sdensity.loc == jnp.array(0.0)
    assert sdensity.scale == jnp.array(1.0)
