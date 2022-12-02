# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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
from typing import Dict, List

import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import pytest
from jax.config import config
from jaxtyping import Array, Float

from jaxlinop import (
    LinearOperator,
    identity,
)

from gpjax.kernels import (
    RBF,
    Linear,
    RationalQuadratic,
    CombinationKernel,
    GraphKernel,
    AbstractKernel,
    Matern12,
    Matern32,
    Matern52,
    Polynomial,
    PoweredExponential,
    ProductKernel,
    Periodic,
    SumKernel,
    _EigenKernel,
    euclidean_distance,
)
from gpjax.parameters import initialise
from gpjax.types import PRNGKeyType

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

        def _initialise_params(self, key: PRNGKeyType) -> Dict:
            return {"test": 1.0}

    # Initialise dummy kernel class and test __call__ and _init_params methods:
    dummy_kernel = DummyKernel()
    assert dummy_kernel._initialise_params(_initialise_key) == {"test": 1.0}
    assert dummy_kernel(jnp.array([1.0]), jnp.array([2.0]), {"test": 2.0}) == 4.0


@pytest.mark.parametrize(
    "a, b, distance_to_3dp",
    [
        ([1.0], [-4.0], 5.0),
        ([1.0, -2.0], [-4.0, 3.0], 7.071),
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 2.236),
    ],
)
def test_euclidean_distance(
    a: List[float], b: List[float], distance_to_3dp: float
) -> None:

    # Convert lists to JAX arrays:
    a: Float[Array, "D"] = jnp.array(a)
    b: Float[Array, "D"] = jnp.array(b)

    # Test distance is correct to 3dp:
    assert jnp.round(euclidean_distance(a, b), 3) == distance_to_3dp


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        Matern12(),
        Matern32(),
        Matern52(),
        Linear(),
        Polynomial(),
        RationalQuadratic(),
    ],
)
@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("n", [1, 2, 10])
def test_gram(kernel: AbstractKernel, dim: int, n: int) -> None:

    # Gram constructor static method:
    gram = kernel.gram

    # Inputs x:
    x = jnp.linspace(0.0, 1.0, n * dim).reshape(n, dim)

    # Default kernel parameters:
    params = kernel._initialise_params(_initialise_key)

    # Test gram matrix:
    Kxx = gram(kernel, params, x)
    assert isinstance(Kxx, LinearOperator)
    assert Kxx.shape == (n, n)


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        Matern12(),
        Matern32(),
        Matern52(),
        Linear(),
        Polynomial(),
        RationalQuadratic(),
    ],
)
@pytest.mark.parametrize("num_a", [1, 2, 5])
@pytest.mark.parametrize("num_b", [1, 2, 5])
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_cross_covariance(
    kernel: AbstractKernel, num_a: int, num_b: int, dim: int
) -> None:

    # Cross covariance constructor static method:
    cross_cov = kernel.cross_covariance

    # Inputs a, b:
    a = jnp.linspace(-1.0, 1.0, num_a * dim).reshape(num_a, dim)
    b = jnp.linspace(3.0, 4.0, num_b * dim).reshape(num_b, dim)

    # Default kernel parameters:
    params = kernel._initialise_params(_initialise_key)

    # Test cross covariance, Kab:
    Kab = cross_cov(kernel, params, a, b)
    assert isinstance(Kab, jnp.ndarray)
    assert Kab.shape == (num_a, num_b)


@pytest.mark.parametrize("kernel", [RBF(), Matern12(), Matern32(), Matern52()])
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_call(kernel: AbstractKernel, dim: int) -> None:

    # Datapoint x and datapoint y:
    x = jnp.array([[1.0] * dim])
    y = jnp.array([[0.5] * dim])

    # Defualt parameters:
    params = kernel._initialise_params(_initialise_key)

    # Test calling gives an autocovariance value of no dimension between the inputs:
    kxy = kernel(params, x, y)

    assert isinstance(kxy, jnp.DeviceArray)
    assert kxy.shape == ()


@pytest.mark.parametrize("kern", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("ell, sigma", [(0.1, 0.2), (0.5, 0.1), (0.1, 0.5), (0.5, 0.5)])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_pos_def(
    kern: AbstractKernel, dim: int, ell: float, sigma: float, n: int
) -> None:
    kern = kern(active_dims=list(range(dim)))
    # Gram constructor static method:
    gram = kern.gram

    # Create inputs x:
    x = jr.uniform(_initialise_key, (n, dim))
    params = {"lengthscale": jnp.array([ell]), "variance": jnp.array([sigma])}

    # Test gram matrix eigenvalues are positive:
    Kxx = gram(kern, params, x)
    Kxx += identity(n) * _jitter
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert (eigen_values > 0.0).all()


@pytest.mark.parametrize("kern", [Linear, Polynomial])
@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("shift", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.5])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_pos_def_lin_poly(
    kern: AbstractKernel, dim: int, shift: float, sigma: float, n: int
) -> None:
    kern = kern(active_dims=list(range(dim)))
    # Gram constructor static method:
    gram = kern.gram

    # Create inputs x:
    x = jr.uniform(_initialise_key, (n, dim))
    params = {"variance": jnp.array([sigma]), "shift": jnp.array([shift])}

    # Test gram matrix eigenvalues are positive:
    Kxx = gram(kern, params, x)
    Kxx += I(n) * _jitter
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert (eigen_values > 0.0).all()


@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("ell, sigma", [(0.1, 0.2), (0.5, 0.1), (0.1, 0.5), (0.5, 0.5)])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_pos_def_rq(dim: int, ell: float, sigma: float, alpha: float, n: int) -> None:
    kern = RationalQuadratic(active_dims=list(range(dim)))
    # Gram constructor static method:
    gram = kern.gram

    # Create inputs x:
    x = jr.uniform(_initialise_key, (n, dim))
    params = {
        "lengthscale": jnp.array([ell]),
        "variance": jnp.array([sigma]),
        "alpha": jnp.array([alpha]),
    }

    # Test gram matrix eigenvalues are positive:
    Kxx = gram(kern, params, x)
    Kxx += I(n) * _jitter
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
    gram = kern.gram

    # Create inputs x:
    x = jr.uniform(_initialise_key, (n, dim))
    params = {
        "lengthscale": jnp.array([ell]),
        "variance": jnp.array([sigma]),
        "power": jnp.array([power]),
    }

    # Test gram matrix eigenvalues are positive:
    Kxx = gram(kern, params, x)
    Kxx += I(n) * _jitter
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
    gram = kern.gram

    # Create inputs x:
    x = jr.uniform(_initialise_key, (n, dim))
    params = {
        "lengthscale": jnp.array([ell]),
        "variance": jnp.array([sigma]),
        "period": jnp.array([period]),
    }

    # Test gram matrix eigenvalues are positive:
    Kxx = gram(kern, params, x)
    Kxx += I(n) * _jitter
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
        params = kern._initialise_params(_initialise_key)

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
        Linear,
        Polynomial,
        RationalQuadratic,
        PoweredExponential,
        Periodic,
    ],
)
def test_dtype(kernel: AbstractKernel) -> None:
    parameter_state = initialise(kernel(), _initialise_key)
    params, *_ = parameter_state.unpack()
    for k, v in params.items():
        assert v.dtype == jnp.float64
        assert isinstance(k, str)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("variance", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("shift", [1e-6, 0.1, 1.0])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_polynomial(
    degree: int, dim: int, variance: float, shift: float, n: int
) -> None:

    # Define inputs
    x = jnp.linspace(0.0, 1.0, n * dim).reshape(n, dim)

    # Define kernel
    kern = Polynomial(degree=degree, active_dims=[i for i in range(dim)])

    # Unpack kernel computation
    gram = kern.gram

    # Check name
    assert kern.name == f"Polynomial Degree: {degree}"

    # Initialise parameters
    params = kern._initialise_params(_initialise_key)
    params["shift"] * shift
    params["variance"] * variance

    # Check parameter keys
    assert list(params.keys()) == ["shift", "variance"]

    # Compute gram matrix
    Kxx = gram(kern, params, x)

    # Check shapes
    assert Kxx.shape[0] == x.shape[0]
    assert Kxx.shape[0] == Kxx.shape[1]

    # Test positive definiteness
    Kxx += identity(n) * _jitter
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert (eigen_values > 0).all()


@pytest.mark.parametrize(
    "kernel", [RBF, Matern12, Matern32, Matern52, Linear, Polynomial, RationalQuadratic]
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

        # Unpack kernel computation
        ad_gram = ad_kern.gram
        manual_gram = manual_kern.gram

        # Get initial parameters
        ad_params = ad_kern._initialise_params(_initialise_key)
        manual_params = manual_kern._initialise_params(_initialise_key)

        # Compute gram matrices
        ad_Kxx = ad_gram(ad_kern, ad_params, x)
        manual_Kxx = manual_gram(manual_kern, manual_params, slice)

        # Test gram matrices are equal
        assert jnp.all(ad_Kxx.to_dense() == manual_Kxx.to_dense())


@pytest.mark.parametrize("combination_type", [SumKernel, ProductKernel])
@pytest.mark.parametrize(
    "kernel", [RBF, RationalQuadratic, Linear, Matern12, Matern32, Matern52, Polynomial]
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

    # Unpack kernel computation
    gram = combination_kernel.gram

    # Initialise default parameters
    params = combination_kernel._initialise_params(_initialise_key)

    # Check params are a list of dictionaries
    assert len(params) == n_kerns

    for p in params:
        assert isinstance(p, dict)

    # Check combination kernel set
    assert len(combination_kernel.kernel_set) == n_kerns
    assert isinstance(combination_kernel.kernel_set, list)
    assert isinstance(combination_kernel.kernel_set[0], AbstractKernel)

    # Compute gram matrix
    Kxx = gram(combination_kernel, params, x)

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

    # Unpack kernel computation
    gram = sum_kernel.gram

    # Initialise default parameters
    params = sum_kernel._initialise_params(_initialise_key)

    # Compute gram matrix
    Kxx = gram(sum_kernel, params, x)

    # NOW we do the same thing manually and check they are equal:

    # Unpack kernel computation
    k1_gram = k1.gram
    k2_gram = k2.gram

    # Initialise default parameters
    k1_params = k1._initialise_params(_initialise_key)
    k2_params = k2._initialise_params(_initialise_key)

    # Compute gram matrix
    Kxx_k1 = k1_gram(k1, k1_params, x)
    Kxx_k2 = k2_gram(k2, k2_params, x)

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

    # Unpack kernel computation
    gram = prod_kernel.gram

    # Initialise default parameters
    params = prod_kernel._initialise_params(_initialise_key)

    # Compute gram matrix
    Kxx = gram(prod_kernel, params, x)

    # NOW we do the same thing manually and check they are equal:

    # Unpack kernel computation
    k1_gram = k1.gram
    k2_gram = k2.gram

    # Initialise default parameters
    k1_params = k1._initialise_params(_initialise_key)
    k2_params = k2._initialise_params(_initialise_key)

    # Compute gram matrix
    Kxx_k1 = k1_gram(k1, k1_params, x)
    Kxx_k2 = k2_gram(k2, k2_params, x)

    # Check manual and automatic gram matrices are equal
    assert jnp.all(Kxx.to_dense() == Kxx_k1.to_dense() * Kxx_k2.to_dense())


def test_graph_kernel():
    # Create a random graph, G, and verice labels, x,
    n_verticies = 20
    n_edges = 40
    G = nx.gnm_random_graph(n_verticies, n_edges, seed=123)
    x = jnp.arange(n_verticies).reshape(-1, 1)

    # Compute graph laplacian
    L = nx.laplacian_matrix(G).toarray() + jnp.eye(n_verticies) * 1e-12

    # Create graph kernel
    kern = GraphKernel(laplacian=L)
    assert isinstance(kern, GraphKernel)
    assert isinstance(kern, _EigenKernel)
    assert kern.num_vertex == n_verticies
    assert kern.evals.shape == (n_verticies, 1)
    assert kern.evecs.shape == (n_verticies, n_verticies)

    # Unpack kernel computation
    gram = kern.gram

    # Initialise default parameters
    params = kern._initialise_params(_initialise_key)
    assert isinstance(params, dict)
    assert list(sorted(list(params.keys()))) == [
        "lengthscale",
        "smoothness",
        "variance",
    ]

    # Compute gram matrix
    Kxx = gram(kern, params, x)
    assert Kxx.shape == (n_verticies, n_verticies)

    # Check positive definiteness
    Kxx += identity(n_verticies) * _jitter
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert all(eigen_values > 0)


@pytest.mark.parametrize(
    "kernel", [RBF, Matern12, Matern32, Matern52, Polynomial, Linear, RationalQuadratic]
)
def test_combination_kernel_type(kernel: AbstractKernel) -> None:
    prod_kern = kernel() * kernel()
    assert isinstance(prod_kern, ProductKernel)
    assert isinstance(prod_kern, CombinationKernel)

    add_kern = kernel() + kernel()
    assert isinstance(add_kern, SumKernel)
    assert isinstance(add_kern, CombinationKernel)
