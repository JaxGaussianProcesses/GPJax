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

# %% [markdown]
# # Regression with heavy tailed data

# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
import distrax as dx
import blackjax 
import tensorflow_probability.substrates.jax as tfp


from chex import dataclass
from gpjax.likelihoods import AbstractLikelihood, NonConjugate
from typing import Optional, Callable
import distrax as dx

key = jr.PRNGKey(123)


# %% [markdown]
# ## Specify likelihood

# %%
@dataclass(repr=False)
class StudentT(AbstractLikelihood, NonConjugate):
    """t-distribution likelihood object."""
    name: Optional[str] = "StudentT"

    @property
    def params(self) -> dict:
        return {"obs_noise": jnp.array([1.0]),
        "df": jnp.array([1.0]),
        }

    @property
    def link_function(self) -> Callable:
        def link_fn(x, params: dict) -> dx.Distribution:
            tfp_dist = tfp.substrates.jax.distributions.StudentT(df=params["df"], loc=x, scale=params["obs_noise"])
            return dx.Independent(tfp_dist)
        return link_fn
    
    def predict(self, dist: dx.Distribution, params: dict) -> dx.Distribution:
        pass


# %% [markdown]
# ## Define Gaussian process

# %%
prior = gpx.Prior(kernel = gpx.RBF())
likelihood = StudentT(num_datapoints=42)
posterior = prior * likelihood
print(type(posterior))

# %%
