from itertools import permutations
from chex import assert_equal

import jax.numpy as jnp
import jax.random as jr
from numpy import isin, sort
import pytest
import networkx as nx
from gpjax import kernels
from gpjax.kernels import (
    RBF,
    Kernel,
    Matern12,
    Matern32,
    Matern52,
    Polynomial,
    ProductKernel,
    SumKernel,
    _EigenKernel,
    GraphKernel,
    cross_covariance,
    euclidean_distance,
    gram,
)
from gpjax.parameters import initialise
from gpjax.utils import I


@pytest.mark.parametrize("kern", [RBF(), Matern12(), Matern32(), Matern52()])
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_gram(kern, dim):
    x = jnp.linspace(-1.0, 1.0, num=10).reshape(-1, 1)
    if dim > 1:
        x = jnp.hstack([x] * dim)
    params, _, _ = initialise(kern)
    gram_matrix = gram(kern, x, params)
    assert gram_matrix.shape[0] == x.shape[0]
    assert gram_matrix.shape[0] == gram_matrix.shape[1]


@pytest.mark.parametrize("kern", [RBF(), Matern12(), Matern32(), Matern52()])
@pytest.mark.parametrize("n1", [3, 10, 20])
@pytest.mark.parametrize("n2", [3, 10, 20])
def test_cross_covariance(kern, n1, n2):
    x1 = jnp.linspace(-1.0, 1.0, num=n1).reshape(-1, 1)
    x2 = jnp.linspace(-1.0, 1.0, num=n2).reshape(-1, 1)
    params, _, _ = initialise(kern)
    kernel_matrix = cross_covariance(kern, x1, x2, params)
    assert kernel_matrix.shape == (n1, n2)


@pytest.mark.parametrize("kernel", [RBF(), Matern12(), Matern32(), Matern52()])
def test_call(kernel):
    params, _, _ = initialise(kernel)
    x, y = jnp.array([[1.0]]), jnp.array([[0.5]])
    point_corr = kernel(x, y, params)
    assert isinstance(point_corr, jnp.DeviceArray)
    assert point_corr.shape == ()


@pytest.mark.parametrize("kern", [RBF(), Matern12(), Matern32(), Matern52()])
@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("ell, sigma", [(0.1, 0.1), (0.5, 0.1), (0.1, 0.5), (0.5, 0.5)])
def test_pos_def(kern, dim, ell, sigma):
    n = 30
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)
    if dim > 1:
        x = jnp.hstack((x) * dim)
    params = {"lengthscale": jnp.array([ell]), "variance": jnp.array(sigma)}

    gram_matrix = gram(kern, x, params)
    jitter_matrix = I(n) * 1e-6
    gram_matrix += jitter_matrix
    min_eig = jnp.linalg.eigvals(gram_matrix).min()
    assert min_eig > 0


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("dim", [1, 2, 5, 10])
def test_initialisation(kernel, dim):
    kern = kernel(active_dims=[i for i in range(dim)])
    params, _, _ = initialise(kern)
    assert list(params.keys()) == ["lengthscale", "variance"]
    assert all(params["lengthscale"] == jnp.array([1.0] * dim))
    assert params["variance"] == jnp.array([1.0])
    if dim > 1:
        assert kern.ard
    else:
        assert not kern.ard


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
def test_dtype(kernel):
    params, _, _ = initialise(kernel())
    for k, v in params.items():
        assert v.dtype == jnp.float64


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("dim", [1, 2, 5])
@pytest.mark.parametrize("variance", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("shift", [1e-6, 0.1, 1.0])
def test_polynomial(degree, dim, variance, shift):
    x = jnp.linspace(0.0, 1.0, num=20).reshape(-1, 1)
    if dim > 1:
        x = jnp.hstack([x] * dim)
    kern = Polynomial(degree=degree, active_dims=[i for i in range(dim)])
    params = kern.params
    params["shift"] * shift
    params["variance"] * variance
    gram_matrix = gram(kern, x, params)
    assert kern.name == f"Polynomial Degree: {degree}"
    jitter_matrix = I(20) * 1e-6
    gram_matrix += jitter_matrix
    min_eig = jnp.linalg.eigvals(gram_matrix).min()
    assert min_eig > 0
    assert gram_matrix.shape[0] == x.shape[0]
    assert gram_matrix.shape[0] == gram_matrix.shape[1]
    assert list(params.keys()) == ["shift", "variance"]


def test_euclidean_distance():
    x1 = jnp.array(1.0)
    x2 = jnp.array(-4.0)
    x1vec = jnp.array((1, 2, 3))
    x2vec = jnp.array((1, 1, 1))
    assert euclidean_distance(x1vec, x2vec) == 3.0
    assert euclidean_distance(x1, x2) == 5.0


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
def test_active_dim(kernel):
    dim_list = [0, 1, 2, 3]
    perm_length = 2
    dim_pairs = list(permutations(dim_list, r=perm_length))
    n_dims = len(dim_list)
    key = jr.PRNGKey(123)
    X = jr.normal(key, shape=(20, n_dims))

    for dp in dim_pairs:
        Xslice = X[..., dp]
        ad_kern = kernel(active_dims=dp)
        manual_kern = kernel(active_dims=[i for i in range(perm_length)])

        k1 = gram(ad_kern, X, ad_kern.params)
        k2 = gram(manual_kern, Xslice, manual_kern.params)
        assert jnp.all(k1 == k2)


@pytest.mark.parametrize("combination_type", [SumKernel, ProductKernel])
@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52, Polynomial])
@pytest.mark.parametrize("n_kerns", [2, 3, 4])
def test_combination_kernel(combination_type, kernel, n_kerns):
    n = 20
    kern_list = [kernel() for _ in range(n_kerns)]
    c_kernel = combination_type(kernel_set=kern_list)
    assert len(c_kernel.kernel_set) == n_kerns
    assert len(c_kernel.params) == n_kerns
    assert isinstance(c_kernel.kernel_set, list)
    assert isinstance(c_kernel.kernel_set[0], Kernel)
    assert isinstance(c_kernel.params[0], dict)
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)
    Kff = gram(c_kernel, x, c_kernel.params)
    assert Kff.shape[0] == Kff.shape[1]
    assert Kff.shape[1] == n


@pytest.mark.parametrize("k1", [RBF(), Matern12(), Matern32(), Matern52(), Polynomial()])
@pytest.mark.parametrize("k2", [RBF(), Matern12(), Matern32(), Matern52(), Polynomial()])
def test_sum_kern_value(k1, k2):
    n = 10
    sum_kernel = SumKernel(kernel_set=[k1, k2])
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)
    Kff = gram(sum_kernel, x, sum_kernel.params)
    Kff_manual = gram(k1, x, k1.params) + gram(k2, x, k2.params)
    assert jnp.all(Kff == Kff_manual)


@pytest.mark.parametrize("k1", [RBF(), Matern12(), Matern32(), Matern52(), Polynomial()])
@pytest.mark.parametrize("k2", [RBF(), Matern12(), Matern32(), Matern52(), Polynomial()])
def test_prod_kern_value(k1, k2):
    n = 10
    sum_kernel = ProductKernel(kernel_set=[k1, k2])
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)
    Kff = gram(sum_kernel, x, sum_kernel.params)
    Kff_manual = gram(k1, x, k1.params) * gram(k2, x, k2.params)
    assert jnp.all(Kff == Kff_manual)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52, Polynomial])
def test_update_params(kernel):
    kern = kernel()
    params = kern.params
    for k in params.keys():
        params[k] = 100
        kern._params = params
        assert kern.params == params


def test_graph_kernel():
    n_verticies = 20
    n_edges = 40
    G = nx.gnm_random_graph(n_verticies, n_edges, seed=123)
    L = nx.laplacian_matrix(G).toarray() + jnp.eye(n_verticies) * 1e-12
    kern = GraphKernel(laplacian=L)
    assert isinstance(kern, GraphKernel)
    assert isinstance(kern, _EigenKernel)

    kern_params = kern.params
    assert isinstance(kern_params, dict)
    assert list(sorted(list(kern_params.keys()))) == ["lengthscale", "smoothness", "variance"]
    x = jnp.arange(n_verticies).reshape(-1, 1)
    Kxx = gram(kern, x, kern.params)
    assert Kxx.shape == (n_verticies, n_verticies)
    kevals, kevecs = jnp.linalg.eigh(Kxx + jnp.eye(n_verticies) * 1e-8)
    assert all(kevals > 0)

    assert kern.num_vertex == n_verticies
    assert kern.evals.shape == (n_verticies, 1)
    assert kern.evecs.shape == (n_verticies, n_verticies)
