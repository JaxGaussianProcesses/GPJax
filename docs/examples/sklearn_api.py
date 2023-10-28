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
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: gpjax
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SKLearn-API
#
# In this notebook we demonstrate the high-level API that we provide in GPJax that is designed to be similar to the API of [scikit-learn](https://scikit-learn.org/stable/).

# %%
from jax.config import config

config.update("jax_enable_x64", True)

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


key = jr.PRNGKey(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Dataset
#
# With the necessary modules imported, we simulate a dataset
# $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y}) = \{(x_i, y_i)\}_{i=1}^{100}$ with inputs $\boldsymbol{x}$
# sampled uniformly on $(-3., 3)$ and corresponding independent noisy outputs
#
# $$\boldsymbol{y} \sim \mathcal{N} \left(\sin(4\boldsymbol{x}) + \cos(2 \boldsymbol{x}), \textbf{I} * 0.3^2 \right).$$
#
# We store our data $\mathcal{D}$ as a GPJax `Dataset` and create test inputs and labels
# for later.

# %%
key = jr.PRNGKey(123)

# Simulate data
n = 200
noise = 0.3
key, subkey = jr.split(key)
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(subkey, shape=signal.shape) * noise
xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
ytest = f(xtest)


fig, ax = plt.subplots()
ax.plot(x, y, "o", label="Observations", color=cols[0])
ax.plot(xtest, ytest, label="Latent function", color=cols[1])
ax.legend(loc="best")

# %% [markdown]
# ## Model building
#
# We'll now proceed to build our model. Within the SKLearn API we have three main classes: `GPJaxRegressor`, `GPJaxClassifier`, and `GPJaxOptimizer`/`GPJaxOptimiser`. We'll consider a problem where the response is continuous and so we'll use the `GPJaxRegressor` class. The problem is identical to the one considered in the [Regression notebook](regression.py); however, we'll now use the SKLearn API to build our model. This offers an alternative to the lower-level API and is designed to be similar to the API of [scikit-learn](https://scikit-learn.org/stable/).

# %%
model = gpx.sklearn.GPJaxRegressor(kernel=gpx.kernels.RBF())
model

# %%
model.fit(x, y, key=key)

# %%
from sklearn.metrics import mean_squared_error

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder & Daniel Dodd'
