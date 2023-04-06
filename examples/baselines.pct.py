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

# %%
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax import jit, grad
from jax.config import config

import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
key = jr.PRNGKey(123)

# %%
n = 1000
noise = 0.3

key, subkey = jr.split(key)
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(subkey, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

# %%
kernel = gpx.kernels.RBF()
meanf = gpx.mean_functions.Constant(constant=0.0)
meanf = meanf.replace_trainable(constant=False)
prior = gpx.Prior(mean_function=meanf, kernel=kernel)
likelihood = gpx.Gaussian(num_datapoints=D.n)

posterior = prior * likelihood

negative_mll = gpx.objectives.ConjugateMLL(negative=True)

# %timeit negative_mll(posterior, train_data=D).block_until_ready()

# %%
# %timeit jit(negative_mll)(posterior, train_data=D).block_until_ready()

# %%
# %timeit grad(negative_mll)(posterior, train_data=D)

# %%
