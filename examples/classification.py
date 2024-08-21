# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: gpjax
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Classification
#
# In this notebook we demonstrate how to perform inference for Gaussian process models
# with non-Gaussian likelihoods via maximum a posteriori (MAP). We focus on a classification task here.

# %%
import cola
from flax import nnx
import jax

# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
import matplotlib.pyplot as plt
import optax as ox
import tensorflow_probability.substrates.jax as tfp

from examples.utils import use_mpl_style
from gpjax.lower_cholesky import lower_cholesky

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


tfd = tfp.distributions
identity_matrix = jnp.eye

# set the default style for plotting
use_mpl_style()

key = jr.key(42)
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Dataset
#
# With the necessary modules imported, we simulate a dataset
# $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y}) = \{(x_i, y_i)\}_{i=1}^{100}$ with inputs
# $\boldsymbol{x}$ sampled uniformly on $(-1., 1)$ and corresponding binary outputs
#
# $$
# \boldsymbol{y} = 0.5 * \text{sign}(\cos(2 *  + \boldsymbol{\epsilon})) + 0.5, \quad \boldsymbol{\epsilon} \sim \mathcal{N} \left(\textbf{0}, \textbf{I} * (0.05)^{2} \right).
# $$
#
# We store our data $\mathcal{D}$ as a GPJax `Dataset` and create test inputs for
# later.

# %%
key, subkey = jr.split(key)
x = jr.uniform(key, shape=(100, 1), minval=-1.0, maxval=1.0)
y = 0.5 * jnp.sign(jnp.cos(3 * x + jr.normal(subkey, shape=x.shape) * 0.05)) + 0.5

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-1.0, 1.0, 500).reshape(-1, 1)

fig, ax = plt.subplots()
ax.scatter(x, y)

# %% [markdown]
# ## MAP inference
#
# We begin by defining a Gaussian process prior with a radial basis function (RBF)
# kernel, chosen for the purpose of exposition. Since our observations are binary, we
# choose a Bernoulli likelihood with a probit link function.

# %%
kernel = gpx.kernels.RBF()
meanf = gpx.mean_functions.Constant()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
likelihood = gpx.likelihoods.Bernoulli(num_datapoints=D.n)

# %% [markdown]
# We construct the posterior through the product of our prior and likelihood.

# %%
posterior = prior * likelihood
print(type(posterior))

# %% [markdown]
# Whilst the latent function is Gaussian, the posterior distribution is non-Gaussian
# since our generative model first samples the latent GP and propagates these samples
# through the likelihood function's inverse link function. This step prevents us from
# being able to analytically integrate the latent function's values out of our
# posterior, and we must instead adopt alternative inference techniques. We begin with
# maximum a posteriori (MAP) estimation, a fast inference procedure to obtain point
# estimates for the latent function and the kernel's hyperparameters by maximising the
# marginal log-likelihood.

# %% [markdown]
# We can obtain a MAP estimate by optimising the log-posterior density with
# Optax's optimisers.

# %%

optimiser = ox.adam(learning_rate=0.01)

opt_posterior, history = gpx.fit(
    model=posterior,
    # we use the negative lpd as we are minimising
    objective=lambda p, d: -gpx.objectives.log_posterior_density(p, d),
    train_data=D,
    optim=ox.adamw(learning_rate=0.01),
    num_iters=1000,
    key=key,
)

# %% [markdown]
# From which we can make predictions at novel inputs, as illustrated below.

# %%
map_latent_dist = opt_posterior.predict(xtest, train_data=D)
predictive_dist = opt_posterior.likelihood(map_latent_dist)

predictive_mean = predictive_dist.mean()
predictive_std = predictive_dist.stddev()

fig, ax = plt.subplots()
ax.scatter(x, y, label="Observations", color=cols[0])
ax.plot(xtest, predictive_mean, label="Predictive mean", color=cols[1])
ax.fill_between(
    xtest.squeeze(),
    predictive_mean - predictive_std,
    predictive_mean + predictive_std,
    alpha=0.2,
    color=cols[1],
    label="One sigma",
)
ax.plot(
    xtest,
    predictive_mean - predictive_std,
    color=cols[1],
    linestyle="--",
    linewidth=1,
)
ax.plot(
    xtest,
    predictive_mean + predictive_std,
    color=cols[1],
    linestyle="--",
    linewidth=1,
)

ax.legend()
# %% [markdown]
# Here we projected the map estimates $\hat{\boldsymbol{f}}$ for the function values
# $\boldsymbol{f}$ at the data points $\boldsymbol{x}$ to get predictions over the
# whole domain,
#
# \begin{align}
# p(f(\cdot)| \mathcal{D})  \approx q_{map}(f(\cdot)) := \int p(f(\cdot)| \boldsymbol{f}) \delta(\boldsymbol{f} - \hat{\boldsymbol{f}}) d \boldsymbol{f} = \mathcal{N}(\mathbf{K}_{\boldsymbol{(\cdot)x}}  \mathbf{K}_{\boldsymbol{xx}}^{-1} \hat{\boldsymbol{f}},  \mathbf{K}_{\boldsymbol{(\cdot, \cdot)}} - \mathbf{K}_{\boldsymbol{(\cdot)\boldsymbol{x}}} \mathbf{K}_{\boldsymbol{xx}}^{-1} \mathbf{K}_{\boldsymbol{\boldsymbol{x}(\cdot)}}).
# \end{align}

# %% [markdown]
# However, as a point estimate, MAP estimation is severely limited for uncertainty
# quantification, providing only a single piece of information about the posterior.

# %% [markdown]
# ## Laplace approximation
# The Laplace approximation improves uncertainty quantification by incorporating
# curvature induced by the marginal log-likelihood's Hessian to construct an
# approximate Gaussian distribution centered on the MAP estimate. Writing
# $\tilde{p}(\boldsymbol{f}|\mathcal{D}) = p(\boldsymbol{y}|\boldsymbol{f}) p(\boldsymbol{f})$
# as the unormalised posterior for function values $\boldsymbol{f}$ at the datapoints
# $\boldsymbol{x}$, we can expand the log of this about the posterior mode
# $\hat{\boldsymbol{f}}$ via a Taylor expansion. This gives:
#
# \begin{align}
# \log\tilde{p}(\boldsymbol{f}|\mathcal{D}) = \log\tilde{p}(\hat{\boldsymbol{f}}|\mathcal{D}) + \left[\nabla \log\tilde{p}({\boldsymbol{f}}|\mathcal{D})|_{\hat{\boldsymbol{f}}}\right]^{T} (\boldsymbol{f}-\hat{\boldsymbol{f}}) + \frac{1}{2} (\boldsymbol{f}-\hat{\boldsymbol{f}})^{T} \left[\nabla^2 \tilde{p}(\boldsymbol{y}|\boldsymbol{f})|_{\hat{\boldsymbol{f}}} \right] (\boldsymbol{f}-\hat{\boldsymbol{f}}) + \mathcal{O}(\lVert \boldsymbol{f} - \hat{\boldsymbol{f}} \rVert^3).
# \end{align}
#
# Since $\nabla \log\tilde{p}({\boldsymbol{f}}|\mathcal{D})$ is zero at the mode,
# this suggests the following approximation
# \begin{align}
# \tilde{p}(\boldsymbol{f}|\mathcal{D}) \approx \log\tilde{p}(\hat{\boldsymbol{f}}|\mathcal{D}) \exp\left\{ \frac{1}{2} (\boldsymbol{f}-\hat{\boldsymbol{f}})^{T} \left[-\nabla^2 \tilde{p}(\boldsymbol{y}|\boldsymbol{f})|_{\hat{\boldsymbol{f}}} \right] (\boldsymbol{f}-\hat{\boldsymbol{f}}) \right\}
# \end{align},
#
# that we identify as a Gaussian distribution,
# $p(\boldsymbol{f}| \mathcal{D}) \approx q(\boldsymbol{f}) := \mathcal{N}(\hat{\boldsymbol{f}}, [-\nabla^2 \tilde{p}(\boldsymbol{y}|\boldsymbol{f})|_{\hat{\boldsymbol{f}}} ]^{-1} )$.
# Since the negative Hessian is positive definite, we can use the Cholesky
# decomposition to obtain the covariance matrix of the Laplace approximation at the
# datapoints below.

# %%


gram, cross_covariance = (kernel.gram, kernel.cross_covariance)
jitter = 1e-6

# Compute (latent) function value map estimates at training points:
Kxx = opt_posterior.prior.kernel.gram(x)
Kxx += identity_matrix(D.n) * jitter
Kxx = cola.PSD(Kxx)
Lx = lower_cholesky(Kxx)
f_hat = Lx @ opt_posterior.latent.value

# Negative Hessian,  H = -∇²p_tilde(y|f):
graphdef, params, *static_state = nnx.split(
    opt_posterior, gpx.parameters.Parameter, ...
)


def loss(params, D):
    model = nnx.merge(graphdef, params, *static_state)
    return -gpx.objectives.log_posterior_density(model, D)


jacobian = jax.jacfwd(jax.jacrev(loss))(params, D)
H = jacobian["latent"].value["latent"].value[:, 0, :, 0]
L = jnp.linalg.cholesky(H + identity_matrix(D.n) * jitter)

# H⁻¹ = H⁻¹ I = (LLᵀ)⁻¹ I = L⁻ᵀL⁻¹ I
L_inv = jsp.linalg.solve_triangular(L, identity_matrix(D.n), lower=True)
H_inv = jsp.linalg.solve_triangular(L.T, L_inv, lower=False)
LH = jnp.linalg.cholesky(H_inv)
laplace_approximation = tfd.MultivariateNormalTriL(f_hat.squeeze(), LH)


# %% [markdown]
# For novel inputs, we must project the above approximating distribution through the
# Gaussian conditional distribution $p(f(\cdot)| \boldsymbol{f})$,
#
# \begin{align}
# p(f(\cdot)| \mathcal{D}) \approx q_{Laplace}(f(\cdot)) := \int p(f(\cdot)| \boldsymbol{f}) q(\boldsymbol{f}) d \boldsymbol{f} = \mathcal{N}(\mathbf{K}_{\boldsymbol{(\cdot)x}}  \mathbf{K}_{\boldsymbol{xx}}^{-1} \hat{\boldsymbol{f}},  \mathbf{K}_{\boldsymbol{(\cdot, \cdot)}} - \mathbf{K}_{\boldsymbol{(\cdot)\boldsymbol{x}}} \mathbf{K}_{\boldsymbol{xx}}^{-1} (\mathbf{K}_{\boldsymbol{xx}} - [-\nabla^2 \tilde{p}(\boldsymbol{y}|\boldsymbol{f})|_{\hat{\boldsymbol{f}}} ]^{-1}) \mathbf{K}_{\boldsymbol{xx}}^{-1} \mathbf{K}_{\boldsymbol{\boldsymbol{x}(\cdot)}}).
# \end{align}
#
# This is the same approximate distribution $q_{map}(f(\cdot))$, but we have perturbed
# the covariance by a curvature term of
# $\mathbf{K}_{\boldsymbol{(\cdot)\boldsymbol{x}}} \mathbf{K}_{\boldsymbol{xx}}^{-1} [-\nabla^2 \tilde{p}(\boldsymbol{y}|\boldsymbol{f})|_{\hat{\boldsymbol{f}}} ]^{-1} \mathbf{K}_{\boldsymbol{xx}}^{-1} \mathbf{K}_{\boldsymbol{\boldsymbol{x}(\cdot)}}$.
# We take the latent distribution computed in the previous section and add this term
# to the covariance to construct $q_{Laplace}(f(\cdot))$.


# %%
def construct_laplace(test_inputs: Float[Array, "N D"]) -> tfd.MultivariateNormalTriL:
    map_latent_dist = opt_posterior.predict(xtest, train_data=D)

    Kxt = opt_posterior.prior.kernel.cross_covariance(x, test_inputs)
    Kxx = opt_posterior.prior.kernel.gram(x)
    Kxx += identity_matrix(D.n) * jitter
    Kxx = cola.PSD(Kxx)

    # Kxx⁻¹ Kxt
    Kxx_inv_Kxt = cola.solve(Kxx, Kxt)

    # Ktx Kxx⁻¹[ H⁻¹ ] Kxx⁻¹ Kxt
    laplace_cov_term = jnp.matmul(jnp.matmul(Kxx_inv_Kxt.T, H_inv), Kxx_inv_Kxt)

    mean = map_latent_dist.mean()
    covariance = map_latent_dist.covariance() + laplace_cov_term
    L = jnp.linalg.cholesky(covariance)
    return tfd.MultivariateNormalTriL(jnp.atleast_1d(mean.squeeze()), L)


# %% [markdown]
# From this we can construct the predictive distribution at the test points.
# %%
laplace_latent_dist = construct_laplace(xtest)
predictive_dist = opt_posterior.likelihood(laplace_latent_dist)

predictive_mean = predictive_dist.mean()
predictive_std = predictive_dist.stddev()

fig, ax = plt.subplots()
ax.scatter(x, y, label="Observations", color=cols[0])
ax.plot(xtest, predictive_mean, label="Predictive mean", color=cols[1])
ax.fill_between(
    xtest.squeeze(),
    predictive_mean - predictive_std,
    predictive_mean + predictive_std,
    alpha=0.2,
    color=cols[1],
    label="One sigma",
)
ax.plot(
    xtest,
    predictive_mean - predictive_std,
    color=cols[1],
    linestyle="--",
    linewidth=1,
)
ax.plot(
    xtest,
    predictive_mean + predictive_std,
    color=cols[1],
    linestyle="--",
    linewidth=1,
)
ax.legend()

# %% [markdown]
# ## System configuration

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -a "Thomas Pinder & Daniel Dodd"
