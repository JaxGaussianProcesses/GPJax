# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Sparse Regression
#
# In this notebook we'll demonstrate how the sparse variational Gaussian process model of <strong data-cite="hensman2013gaussian"></strong>. When seeking to model more than ~5000 data points and/or the assumed likelihood is non-Gaussian, the sparse Gaussian process presented here will be a tractable option. However, for models of less than 5000 data points and a Gaussian likelihood function, we would recommend using the marginal log-likelihood approach presented in [Regression](https://gpjax.readthedocs.io/en/latest/nbs/regression.html).

# %%
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import jit
from jax.experimental import optimizers

key = jr.PRNGKey(123)

# %% [markdown]
# ## Data
#
# We'll simulate 5000 observation inputs $X$ and simulate the corresponding output $y$ according to 
# $$y = \sin(4X) + \cos(2X)\,.$$
# We'll perturb our observed responses through a sequence independent and identically distributed draws from $\mathcal{0, 0.2}$.
#

# %%
N = 5000
noise = 0.2

x = jr.uniform(key=key, minval=-5.0, maxval=5.0, shape=(N,)).sort().reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise
xtest = jnp.linspace(-6.0, 6.0, 500).reshape(-1, 1)

# %% [markdown]
# ## Inducing points
#
# Tractability in a sparse Gaussian process is made possible through a set of inducing points $Z$. At a high-level, the set of inducing points acts as a pseudo-dataset that enables low-rank approximations $\mathbf{K}_{zz}$ of the true covariance matrix $\mathbf{K}_{xx}$ to be computed. More tricks involving a variational treatment of the model's marginal log-likelihood unlock the full power of sparse GPs, but more on that later. For now, we'll initialise a set of inducing points using a linear spaced grid across our observed data's support.

# %%
Z = jnp.linspace(-5.0, 5.0, 50).reshape(-1, 1)

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", alpha=0.3)
ax.plot(xtest, f(xtest))
ax.scatter(Z, jnp.zeros_like(Z), marker="|", color="black")
[ax.axvline(x=z, color="black", alpha=0.3, linewidth=1) for z in Z]
plt.show()

# %% [markdown]
# ## Defining processes
#
# Unlike regular GP regression, we won't ever acess the marginal log-likelihood of our true process. Instead, we'll introduce a variational approximation $q$ that is itself a Gaussian process. We'll then seek to minimise the Kullback-Leibler divergence $\operatorname{KL}(\cdot || \cdot)$ from our approximate process $q$ to the true process $p$ through the evidence lower bound. 

# %%
D = gpx.Dataset(X=x, y=y)
true_process = gpx.Prior(kernel=gpx.RBF()) * gpx.Gaussian(num_datapoints=N)

q = gpx.VariationalGaussian(inducing_inputs=Z)

# %%
svgp = gpx.SVGP(posterior=true_process, variational_family=q)

# %%
params, trainables, constrainers, unconstrainers = gpx.initialise(svgp)
params = gpx.transform(params, unconstrainers)

loss_fn = jit(gpx.VFE(svgp, D, constrainers))

# %%
opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)

batched_dataset = jit(gpx.abstractions.mini_batcher(D, batch_size=256, prefetch_buffer=1))

learned_params = gpx.abstractions.fit_batches(
    loss_fn,
    params,
    trainables,
    opt_init,
    opt_update,
    get_params,
    get_batch=batched_dataset,
    n_iters=2500,
    jit_compile=True,
)
learned_params = gpx.transform(learned_params, constrainers)

# %%
meanf = svgp.mean(learned_params)(xtest)
varfs = jnp.diag(svgp.variance(learned_params)(xtest))
meanf = meanf.squeeze()
varfs = varfs.squeeze() + learned_params['likelihood']['obs_noise']

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", alpha=0.15, label="Training Data", color="tab:gray")
ax.plot(xtest, meanf, label="Posterior mean", color="tab:blue")
ax.fill_between(xtest.flatten(), meanf - jnp.sqrt(varfs), meanf + jnp.sqrt(varfs), alpha=0.3)
[ax.axvline(x=z, color="black", alpha=0.3, linewidth=1) for z in learned_params['variational_family']['inducing_inputs']]
plt.show()

# %%
import matplotlib.pyplot as plt
import jax.numpy as jnp 
import gpjax as gpx 
import tensorflow_probability.substrates.jax as tfp
tfb = tfp.bijectors 

x = jnp.linspace(-5., 5., 3).reshape(-1, 1)
kern=gpx.kernels.RBF()
Kxx = gpx.kernels.gram(kern, x, kern.params)
plt.matshow(Kxx)

# %%
L = jnp.linalg.cholesky(Kxx)
L

# %%
bij = tfb.FillTriangular()
Lb = bij.inverse(L)
Lb

# %%
bij.inverse_log_det_jacobian(L)

# %%
bij.forward(Lb) - L

# %%
bij.forward_log_det_jacobian(Lb)

# %%
n = L.shape[0]
x = jnp.zeros_like(Kxx)
# Get the indexes for which we need to fill the triangular matrix
idxs = jnp.tril_indices(n, 0)
# Fill the triangular matrix
xT = x.at[idxs].set(Lb)
xT

# %%
xT[idxs]

# %%
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Fill triangular bijector."""

import jax.numpy as jnp
from typing import Tuple, Optional

from distrax._src.bijectors import bijector as base

Array = base.Array


class FillTriangular(base.Bijector):
  """A transformation that maps a vector to a triangular matrix. The triangular
   matrix can be either upper or lower triangular. By default, the lower
   triangular matrix is used.

  When projecting from a vector to a triangular matrix, entries of the matrix
   are populated row-wise. For example, if the vector is [1, 2, 3, 4, 5, 6],
   the triangular matrix will be:
  [[1, 0, 0],
  [2, 3, 0],
  [4, 5, 6]].
  """

  def __init__(
    self,
    is_lower: Optional[bool] = True,
  ):
    """Initialise the `FillTriangular` bijector.

  Args:
  matrix_shape (int): The number of rows (or columns) in the original
  triangular matrix.
  upper (Optional[bool]): Whether or not the matrix being transformed
  is an upper or lower-triangular matrix. Defaults to True.
  """ """"""
    super().__init__(event_ndims_in=0)
    self.index_fn = jnp.tril_indices if is_lower else jnp.triu_indices

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """The forward method maps from a vector to a triangular matrix.

    Args:
      x (Array): The 1-dimensional vector that is to be mapped into a
      triangular matrix.

    Returns:
      Tuple[Array, Array]: A triangular matrix and the log determinant of the
      Jacobian. The log-determinant here is just 0. as the bijection is simply
      reshaping.
    """
    vector_length = x.shape[0]
    matrix_shape = (jnp.sqrt(0.25 + 2. * vector_length) - 0.5).astype(int)
    y = jnp.zeros((matrix_shape, matrix_shape))
    # Get the indexes for which we need to fill the triangular matrix
    idxs = self.index_fn(matrix_shape)
    # Fill the triangular matrix
    y = y.at[idxs].set(x)
    return y, jnp.array(0.0)

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """The inverse method maps from a triangular matrix to a vector.

    Args:
      y (Array): The lower triangular

    Returns:
      Tuple[Array, Array]: The vectorised form of the supplied triangular
      matrix and the log determinant of the Jacobian. The log-determinant
      here is just 0. as the bijection is simply reshaping.
    """
    matrix_shape = y.shape[0]
    return y[self.index_fn(matrix_shape)], jnp.array(0.0)



dbij = FillTriangular(is_lower=True)
Ldb = dbij.inverse(L) # Matrix to vector
Ldb

# %%
dbij.forward(Ldb) - L

# %%
dbij = FillTriangular(matrix_shape=3, upper=True)
Ldb = dbij.forward(L.T)
assert (dbij.inverse(Ldb) - L.T == 0).all()

# %%
Ldb

# %%
xvec = jnp.array([1,2,3,4,5,6])
jnp.triu(dbij.inverse(xvec)) == xvec


# %%
