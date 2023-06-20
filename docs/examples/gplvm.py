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
#     display_name: gpjax_beartype
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gaussian Process Latent Variable Models
#
# The Gaussian process latent variable model (GPLVM)
# <strong data-cite="lawrence2003gaussian"></strong> employs GPs to learn a
# low-dimensional latent space representation of a high-dimensional, unsupervised
# dataset.

# %%

from jax.config import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib as mpl
from jaxtyping import install_import_hook
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from dataclasses import dataclass
import optax as ox
import seaborn as sns

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.base import param_field
    from gpjax.typing import ScalarFloat

key = jr.PRNGKey(123)
plt.style.use("./gpjax.mplstyle")
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
#
# Using the [countries of the world](https://www.kaggle.com/datasets/fernandol/countries-of-the-world) data from [Kaggle](https://www.kaggle.com/).
#
# +
# Observed temperature data

# %%

try:
    world_data = pd.read_csv("data/countries_of_the_world.csv", decimal=",").dropna()
except FileNotFoundError:
    world_data = pd.read_csv(
        "docs/examples/data/countries_of_the_world.csv", decimal=","
    ).dropna()

world_data.head()

label = world_data[["GDP ($ per capita)"]]
features = world_data.drop(["Country", "Region", "GDP ($ per capita)"], axis="columns")

# %% [markdown]
# ### Parameters
#
# To aid inference in our model, we'll initialise the latent coordinates using
# principal component analysis.

# %%
features = StandardScaler().fit_transform(features)
latent_dim = 2
principal_components = PCA(n_components=latent_dim).fit_transform(features)
initial_X = jnp.asarray(principal_components)

# %% [markdown]
#
# ## Model specification
#
# GPLVMs use a set of $Q$ Gaussian process $(f_1, f_2, \ldots, f_Q)$ to project from
# the latent space $\mathbf{X}\in\mathbb{R}^{N\times Q}$ to the observed dataset
# $\mathbf{Y}\in\mathbb{R}^{N\times D}$ where $Q\ll D$. The hierarchical model can then
# be written as
# $$\begin{align}
# p(\mathbf{X}) & = \prod_{n=1}^N \mathcal{N}(\mathbf{x}_{n}\mid\mathbf{0}, \mathbf{I}_Q) \\
# p(\mathbf{f}\mid \mathbf{X}, \mathbf{\theta}) & = \prod_{d=1}^{D} \mathcal{N}(\mathbf{f}_{d}\mid \mathbf{0}, \mathbf{K}_{\mathbf{ff}}) \\
# p(\mathbf{Y}\mid\mathbf{f}, \mathbf{X}) & = \prod_{n=1}^{N}\prod_{d=1}^{D}\mathcal{N}(y_{n, d}\mid f_d(\mathbf{x}_n), \sigma^2)
# \end{align}
# $$
# where $\mathbf{f}_d = f_d(\mathbf{X})$. In the GPLVM implemented with GPJax, we learn
# a MAP estimate of the latent coordinates that enables analytical marginalisation of
# the latent GP.

# %%
meanf = gpx.mean_functions.Zero()
prior = gpx.Prior(
    mean_function=meanf,
    kernel=gpx.kernels.Matern52(active_dims=list(range(latent_dim))),
)
likelihood = gpx.likelihoods.Gaussian(num_datapoints=initial_X.shape[0])

latent_processes = prior * likelihood


# +
@dataclass
class GPLVM(gpx.Module):
    latent_process: gpx.gps.ConjugatePosterior
    latent_variables: jax.Array = param_field()


model = GPLVM(latent_process=latent_processes, latent_variables=initial_X)


# %% [markdown]
# ## Optimisation
#
# We can now maximise the marginal log-likelihood of our GPLVM with respect to the
# kernel parameters, observation noise term, and the latent coordinate. We'll JIT
# compile this function to accelerate optimisation.


# %%
@dataclass
class GPLVM_MAP(gpx.objectives.AbstractObjective):
    def step(
        self,
        posterior: GPLVM,  # noqa: F821
        observed_data: jax.Array,  # noqa: F821
    ) -> ScalarFloat:
        Z = posterior.latent_variables

        def _single_mll(y: jax.Array):
            y = y[:, None]
            D = gpx.Dataset(X=Z, y=y)
            objective = gpx.objectives.ConjugateMLL(negative=True)
            return objective.step(posterior.latent_process, D)

        summed_mll = jnp.sum(jax.vmap(_single_mll)(observed_data))
        return self.constant * summed_mll


objective = jax.jit(GPLVM_MAP().step)
Y = jnp.asarray(features).T

num_iters = 2000

schedule = ox.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=0.1,
    warmup_steps=75,
    decay_steps=num_iters // 2,
    end_value=0.001,
)

optim = ox.adamw(learning_rate=0.01)
state = optim.init(model)


def loss(model) -> ScalarFloat:
    model = model.stop_gradient()
    return objective(model.constrain(), Y)


iter_keys = jr.split(key, num_iters)


def step(carry, key):
    model, opt_state = carry

    loss_val, loss_gradient = jax.value_and_grad(loss)(model)
    updates, opt_state = optim.update(loss_gradient, opt_state, model)
    model = ox.apply_updates(model, updates)

    carry = model, opt_state
    return carry, loss_val


# Optimisation loop.
(model, _), history = gpx.scan.vscan(step, (model, state), (iter_keys), unroll=1)

# Constrained space.
model = model.constrain()

# +
Z = model.latent_variables
results = pd.DataFrame(jnp.hstack((Z, label.values)), columns=["x", "y", "label"])

# %% [markdown]
# ## Latent space visualisation
#
# With optimisation complete, we can now visualise our latent space. To do this, we'll
# simply plot the 2D coordinate that has been learned for each observation and colour
# it by the country's GDP.

# %%
fig, ax = plt.subplots()

scatter = ax.scatter(
    results["x"],
    results["y"],
    c=results["label"],
    cmap="viridis",
    norm=mpl.colors.LogNorm(),
)

legend1 = ax.legend(*scatter.legend_elements(num=6), loc="best", title="Log GDP")
ax.add_artist(legend1)
ax.set(
    xlabel="Latent dimension 1",
    ylabel="Latent dimension 2",
    title="Latent space of GDP data",
)
# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
