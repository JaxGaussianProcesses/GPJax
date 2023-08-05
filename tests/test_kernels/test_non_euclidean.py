# # Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
# #
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

from jax.config import config
import jax.numpy as jnp
import networkx as nx

from gpjax.kernels.non_euclidean import GraphKernel, CatKernel
from gpjax.linops import identity
import jax.random as jr

# # Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


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
    assert kern.num_vertex == n_verticies
    assert kern.eigenvalues.shape == (n_verticies, 1)
    assert kern.eigenvectors.shape == (n_verticies, n_verticies)

    # Compute gram matrix
    Kxx = kern.gram(x)
    assert Kxx.shape == (n_verticies, n_verticies)

    # Check positive definiteness
    Kxx += identity(n_verticies) * 1e-6
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert all(eigen_values > 0)


def test_cat_kernel():
    x = jr.normal(jr.PRNGKey(123), (5000, 3))
    gram = jnp.cov(x.T)
    params = CatKernel.gram_to_sdev_cholesky_lower(gram)
    dk = CatKernel(
        inspace_vals=list(range(len(gram))),
        sdev=params.sdev,
        cholesky_lower=params.cholesky_lower,
    )
    assert jnp.allclose(dk.explicit_gram, gram)

    sdev = jnp.ones((2,))
    cholesky_lower = jnp.eye(2)
    inspace_vals = [0.0, 1.0]

    # Initialize CatKernel object
    dict_kernel = CatKernel(
        sdev=sdev, cholesky_lower=cholesky_lower, inspace_vals=inspace_vals
    )

    assert dict_kernel.sdev.shape == sdev.shape
    assert jnp.allclose(dict_kernel.sdev, sdev)
    assert jnp.allclose(dict_kernel.cholesky_lower, cholesky_lower)
    assert dict_kernel.inspace_vals == inspace_vals


def test_cat_kernel_gram_to_sdev_cholesky_lower():
    gram = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    sdev_expected = jnp.array([1.0, 1.0])
    cholesky_lower_expected = jnp.array([[1.0, 0.0], [0.5, 0.8660254]])

    # Compute sdev and cholesky_lower from gram
    sdev, cholesky_lower = CatKernel.gram_to_sdev_cholesky_lower(gram)

    assert jnp.allclose(sdev, sdev_expected)
    assert jnp.allclose(cholesky_lower, cholesky_lower_expected)


def test_cat_kernel_call():
    sdev = jnp.ones((2,))
    cholesky_lower = jnp.eye(2)
    inspace_vals = [0.0, 1.0]

    # Initialize CatKernel object
    dict_kernel = CatKernel(
        sdev=sdev, cholesky_lower=cholesky_lower, inspace_vals=inspace_vals
    )

    # Compute kernel value for pair of inputs
    kernel_value = dict_kernel.__call__(0, 1)

    assert jnp.allclose(kernel_value, 0.0)  # since cholesky_lower is identity matrix


def test_cat_kernel_explicit_gram():
    sdev = jnp.ones((2,))
    cholesky_lower = jnp.eye(2)
    inspace_vals = [0.0, 1.0]

    # Initialize CatKernel object
    dict_kernel = CatKernel(
        sdev=sdev, cholesky_lower=cholesky_lower, inspace_vals=inspace_vals
    )

    # Compute explicit gram matrix
    explicit_gram = dict_kernel.explicit_gram

    assert explicit_gram.shape == (2, 2)
    assert jnp.allclose(
        explicit_gram, jnp.eye(2)
    )  # since sdev are ones and cholesky_lower is identity matrix
