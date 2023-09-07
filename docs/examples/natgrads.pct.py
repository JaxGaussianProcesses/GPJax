# -*- coding: utf-8 -*-
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
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Natural Gradients

# %% [markdown]
# In this notebook, we show how to create natural gradients. Ordinary gradient descent algorithms are an undesirable for variational inference because we are minimising the KL divergence  between distributions rather than a set of parameters directly. Natural gradients, on the other hand, accounts for the curvature induced by the KL divergence that has the capacity to considerably improve performance (see e.g., <strong data-cite="salimbeni2018">Salimbeni et al. (2018)</strong> for further details).
#
# TODO: Add details about natural gradients from a mathematical perspective.

# %%
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax.config import config
import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
key = jr.PRNGKey(123)

# %% [markdown]
# # Dataset:

# %% [markdown]
# We simulate a dataset $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y}) = \{(x_i, y_i)\}_{i=1}^{5000}$ with inputs $\boldsymbol{x}$ sampled uniformly on $(-5, 5)$ and corresponding binary outputs
#
# $$\boldsymbol{y} \sim \mathcal{N} \left(\sin(4 * \boldsymbol{x}) + \sin(2 * \boldsymbol{x}), \textbf{I} * (0.2)^{2} \right).$$
#
# We store our data $\mathcal{D}$ as a GPJax `Dataset` and create test inputs for later.

# %%
n = 5000
noise = 0.2

x = jr.uniform(key=key, minval=-5.0, maxval=5.0, shape=(n,)).sort().reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-5.5, 5.5, 500).reshape(-1, 1)

# %% [markdown]
# Initialise inducing points:

# %%
z = jnp.linspace(-5.0, 5.0, 20).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", alpha=0.3)
ax.plot(xtest, f(xtest))
[ax.axvline(x=z_i, color="black", alpha=0.3, linewidth=1) for z_i in z]
plt.show()

# %% [markdown]
# # Natural gradients:

# %% [markdown]
# We begin by defining our model and variational family:

# %%
likelihood = gpx.Gaussian(num_datapoints=n)
kernel = gpx.RBF()
prior = gpx.Prior(kernel=kernel, mean_function=gpx.Constant())
p = prior * likelihood
q = gpx.NaturalVariationalGaussian(posterior=p, inducing_inputs=z)

# %%
from jax import grad
import jax.tree_util as jtu
from gpjax.objectives import ELBO


def nat_grad(
    q: gpx.NaturalVariationalGaussian, batch
) -> gpx.NaturalVariationalGaussian:
    # Compute the expectation paramisation.
    exp = q.to_expectation()

    # Compute the natural gradient.
    ss = grad(ELBO(negative=False))(exp, batch)

    # Update the natural parametrisation.
    return q.replace(
        natural_vector=ss.expectation_vector, natural_matrix=ss.expectation_matrix
    )


# %% [markdown]
# Next, we can conduct natural gradients as follows:

# %%
from jax import jit, lax
from gpjax.fit import get_batch


likelihood = gpx.Gaussian(num_datapoints=n)
kernel = gpx.RBF()
prior = gpx.Prior(kernel=kernel, mean_function=gpx.Constant())
p = prior * likelihood
q = gpx.NaturalVariationalGaussian(posterior=p, inducing_inputs=z)

# Define the optimiser:
opt = ox.adam(1e-2)
state = opt.init(q)


# Define the hyperparameter objective:
elbo = ELBO(negative=True)


# Define the E-Step (natural gradient):
def estep(q, state, batch, learning_rate=1.0):
    """Do a natural gradient step"""

    # Compute natural gradient:
    q_nat = nat_grad(q, batch)

    # Compute updates for the dual parameters:
    natural_vector = q.natural_vector + learning_rate * q_nat.natural_vector
    natural_matrix = q.natural_matrix + learning_rate * q_nat.natural_matrix

    # Apply updates:
    q = q.replace(natural_vector=natural_vector, natural_matrix=natural_matrix)
    return q, state


# Define the M-Step (hyperparameter optimisation):
def mstep(q, state, batch):
    """Do a maximisation of the hyperparameters step"""
    g = grad(elbo)(q, batch)
    updates, state = opt.update(g, state, q)
    q_updated = ox.apply_updates(q, updates)
    q = q_updated.replace(
        natural_vector=q.natural_vector, natural_matrix=q.natural_matrix
    )
    return q, state


def step(carry, key):
    key, subkey = jr.split(key)
    batch = get_batch(D, 256, subkey)

    q, state = carry

    q, state = mstep(q, state, batch)
    q, state = estep(q, state, batch)
    return (q, state), elbo(q, D)


(q, _), natural_hist = lax.scan(step, (q, state), jr.split(jr.PRNGKey(42), 100))

q.posterior.likelihood

# %% [markdown]
# Here is the fitted model:

# %%
latent_dist = q(xtest)
predictive_dist = q.posterior.likelihood(latent_dist)

meanf = predictive_dist.mean()
sigma = predictive_dist.stddev()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", alpha=0.15, label="Training Data", color="tab:gray")
ax.plot(xtest, meanf, label="Posterior mean", color="tab:blue")
ax.fill_between(xtest.flatten(), meanf - sigma, meanf + sigma, alpha=0.3)
[ax.axvline(x=z_i, color="black", alpha=0.3, linewidth=1) for z_i in q.inducing_inputs]
plt.show()

# %% [markdown]
# # Natural gradients and sparse varational Gaussian process regression:

# %% [markdown]
# As mentioned in <strong data-cite="hensman2013gaussian">Hensman et al. (2013)</strong>, in the case of a Gaussian likelihood, taking a step of unit length for natural gradients on a full batch of data recovers the same solution as <strong data-cite="titsias2009">Titsias (2009)</strong>. We now illustrate this.

# %% [markdown]
# We begin with natgrads:

# %%
q = gpx.NaturalVariationalGaussian(posterior=p, inducing_inputs=z)
negative_elbo = ELBO(negative=True)


def do_natgrad_step(q, batch, learning_rate=1.0):
    """Do a natural gradient step"""

    # Compute natural gradient:
    q_nat = nat_grad(q, batch)

    # Compute updates for the dual parameters:
    natural_vector = q.natural_vector + learning_rate * q_nat.natural_vector
    natural_matrix = q.natural_matrix + learning_rate * q_nat.natural_matrix

    # Apply updates:
    q = q.replace(natural_vector=natural_vector, natural_matrix=natural_matrix)
    return q


q = do_natgrad_step(q, D)
loss_val = negative_elbo(q, D)
print(loss_val)

# %% [markdown]
# Let us now run it for SGPR:

# %%
q = gpx.CollapsedVariationalGaussian(posterior=p, inducing_inputs=z)

negative_collapsed_elbo = gpx.CollapsedELBO(negative=True)
loss_val = negative_collapsed_elbo(q, D)

print(loss_val)

# %% [markdown]
# We get the same!

# %% [markdown]
# ## DualSVGP

# %% [markdown]
# TODO: Clean up code. Write maths and intuition behind dual SVGP.

# %% [markdown]
# This is not quite right yet.

# %%
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax import jit, grad, lax
from gpjax.objectives import ELBO
import optax as ox
from gpjax.variational_families import DualVariationalGaussian


key = jr.PRNGKey(123)


# Define the variational family:
q = DualVariationalGaussian(posterior=p, inducing_inputs=z)

# Define the optimiser:
opt = ox.adam(1e-2)
state = opt.init(q)

# Define the hyperparameter objective:
elbo = ELBO(negative=True)


@jit
def estep(q, state, batch, learning_rate=1.0):
    """Do a natural gradient step"""

    # Compute natural gradient:
    q_nat = q.natural_gradient(batch)

    # Compute updates for the dual parameters:
    dual_vector = (
        1.0 - learning_rate
    ) * q.dual_vector + learning_rate * q_nat.dual_vector
    dual_matrix = (
        1.0 - learning_rate
    ) * q.dual_matrix + learning_rate * q_nat.dual_matrix

    # Apply updates:
    q = q.replace(dual_vector=dual_vector, dual_matrix=dual_matrix)
    return q, state


@jit
def mstep(q, state, batch):
    """Do a maximisation of the hyperparameters step"""
    g = grad(elbo)(q, batch)
    updates, state = opt.update(g, state, q)
    q_updated = ox.apply_updates(q, updates)
    q = q_updated.replace(dual_vector=q.dual_vector, dual_matrix=q.dual_matrix)
    return q, state


def step(carry, key):
    key, subkey = jr.split(key)
    batch = get_batch(D, 256, subkey)

    q, state = carry
    q, state = mstep(q, state, batch)
    q, state = estep(q, state, batch)

    return (q, state), elbo(q, D)


(q, _), dual_hist = lax.scan(step, (q, state), jr.split(jr.PRNGKey(42), 100))

q.posterior.likelihood

# %%
latent_dist = q(xtest)
predictive_dist = q.posterior.likelihood(latent_dist)

meanf = predictive_dist.mean()
sigma = predictive_dist.stddev()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", alpha=0.15, label="Training Data", color="tab:gray")
ax.plot(xtest, meanf, label="Posterior mean", color="tab:blue")
ax.fill_between(xtest.flatten(), meanf - sigma, meanf + sigma, alpha=0.3)
[ax.axvline(x=z_i, color="black", alpha=0.3, linewidth=1) for z_i in q.inducing_inputs]
plt.show()

# %%
plt.plot(natural_hist, label="Natural")
plt.plot(dual_hist, label="Dual")
plt.legend()

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Daniel Dodd'
