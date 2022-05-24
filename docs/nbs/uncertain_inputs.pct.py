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
import jax.numpy as jnp
import jax.random as jr
import distrax as dx
import optax as ox
import matplotlib.pyplot as plt
from chex import dataclass

key = jr.PRNGKey(123)

# %%
train_x_mean = jnp.linspace(0, 1, 20)
# We'll assume the variance shrinks the closer we get to 1
train_x_stdv = jnp.linspace(0.03, 0.01, 20)

# True function is sin(2*pi*x) with Gaussian noise
train_y = jnp.sin(train_x_mean * (2 * jnp.pi)) + jr.normal(key, shape=train_x_mean.shape) * 0.2

f, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.errorbar(train_x_mean, train_y, xerr=(train_x_stdv * 2), fmt="k*", label="Train Data")
ax.legend()


# %%
@dataclass
class UncertainDataset(gpx.Dataset):
    x_mean = x_mean
    x_std = x_std 
    y = y
    
