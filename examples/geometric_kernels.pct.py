# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.9.7 ('gpjax')
#     language: python
#     name: python3
# ---

# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
from jax.config import config

key = jr.PRNGKey(123)
config.update("jax_enable_x64", True)

# %%
# Create a manifold (2-dim sphere).
hypersphere = Hypersphere(dim=2)

# Generate 3 random points on the sphere.
xs = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

# Initialize kernel, use 100 terms to approximate the infinite series.
kernel = MaternKarhunenLoeveKernel(hypersphere, 10)
params, state = kernel.init_params_and_state()
params["nu"] = jnp.array([5 / 2])
params["lengthscale"] = jnp.array([1.0])

# Compute and print out the 3x3 kernel matrix.
print(kernel.K(params, state, xs))

# %%
from chex import dataclass
from chex import PRNGKey as PRNGKeyType
from typing import Dict
from geometric_kernels.spaces.base import Space
from geometric_kernels.kernels import BaseGeometricKernel
from jaxtyping import Array, Float
import jax


@dataclass(repr=False)
class _GeometricKernel:
    base_kernel: BaseGeometricKernel


@dataclass(repr=False)
class GeometricKernel(
    gpx.kernels.AbstractKernel, gpx.kernels.DenseKernelComputation, _GeometricKernel
):
    def __post_init__(self) -> None:
        _, self.state = self.base_kernel.init_params_and_state()

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        params, _ = self.base_kernel.init_params_and_state()
        params = jax.tree_util.tree_map(jnp.atleast_1d, params)
        return params

    @property
    def space(self) -> Space:
        """Alias to kernel Space"""
        return self._kernel.space

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        kxx = self.base_kernel.K(params, self.state, x, y)
        print(kxx.shape)
        return kxx


# %%
kernel = GeometricKernel(base_kernel=MaternKarhunenLoeveKernel(hypersphere, 10))

# gpx.config.add_parameter("nu", "positive_transform")

# %%
prior = gpx.Prior(kernel=kernel)
# prior = gpx.Prior(kernel=gpx.Matern12(active_dims=[0, 1, 2]))
param_state = gpx.initialise(prior, key=key)

# %%
param_state.params

# %%
print(kernel.state)

# %%
prior.kernel.gram(prior.kernel, param_state.params["kernel"], xs)

# %%
