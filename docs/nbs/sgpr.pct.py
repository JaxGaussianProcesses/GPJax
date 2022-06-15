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
#     display_name: Python 3.9.7 ('gpjax')
#     language: python
#     name: python3
# ---

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Regression
#
# In this notebook we consider sparse Gaussian process regression (SGPR) Titsias 2009. This is a solution for medium- to large-scale conjugate regression problems. 
# In order to arrive at a computationally tractable method, the approximate posterior is parameterized via a set of $m$ pseudo-points \boldsymbol{z}. Critically, the approach leads to $\mathcal{O}(nm^2)$ complexity for approximate maximum likelihood learning and $O(m^2)$ per test point for prediction.

# %%

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax import jit

import gpjax as gpx
import tensorflow as tf

tf.random.set_seed(42)
key = jr.PRNGKey(123)


# %% [markdown]
# ## Dataset
#
# With the necessary modules imported, we simulate a dataset $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y}) = \{(x_i, y_i)\}_{i=1}^{500}$ with inputs $\boldsymbol{x}$ sampled uniformly on $(-3., 3)$ and corresponding independent noisy outputs
#
# $$\boldsymbol{y} \sim \mathcal{N} \left(\sin(7\boldsymbol{x}) + x \cos(2 \boldsymbol{x}), \textbf{I} * 0.5^2 \right).$$
#
# We store our data $\mathcal{D}$ as a GPJax `Dataset` and create test inputs and labels for later.
# %%
# %%
n = 500
noise = .5

x = jr.uniform(key=key, minval=-1.0, maxval=1.0, shape=(n,)).sort().reshape(-1, 1)
f = lambda x: jnp.sin(7 * x) + x * jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-1.1, 1.1, 500).reshape(-1, 1)
ytest = f(xtest)
# %% [markdown]
# To better understand what we have simulated, we plot both the underlying latent function and the observed data that is subject to Gaussian noise. We also plot an initial set of inducing points over the space.
# %%
z = jnp.linspace(-1.0, 1.0, 20).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(xtest, ytest, label="Latent function")
ax.plot(x, y, "o", color="red",  alpha=.8, label="Observations", markersize=2.5)
[ax.axvline(x=z_i, color="black", alpha=0.3, linewidth=1) for z_i in z]
ax.legend(loc="best")
plt.show()
# %% [markdown]
# Next we define the posterior model for the data.
# %%
kernel = gpx.RBF()
likelihood = gpx.Gaussian(num_datapoints=D.n)
prior = gpx.Prior(kernel=kernel)
p = prior * likelihood
# ## 
# %% [markdown]
# We now define the SGPR model.
# %%
q = gpx.CollapsedVariationalGaussian(prior=prior, inducing_inputs=z)

# %% [markdown]
# We define our variational inference algorithm through `CollapsedVI`.
# %%
sgpr = gpx.CollapsedVI(posterior=p, variational_family=q)

# %% [markdown]
# We now train our model.
# %%
params, trainables, constrainers, unconstrainers = gpx.initialise(sgpr)

loss_fn = jit(sgpr.elbo(D, constrainers, negative=True))

optimiser = ox.adam(learning_rate=0.01)

params = gpx.transform(params, unconstrainers)

learned_params = gpx.fit(
    objective = loss_fn,
    params = params,
    trainables = trainables,
    optax_optim = optimiser,
    n_iters=2000,
)
learned_params = gpx.transform(learned_params, constrainers)

# %% [markdown]
# We show predictions.
# %%
latent_dist = q.predict(D, learned_params)(xtest)
predictive_dist = likelihood(latent_dist, learned_params)

samples = latent_dist.sample(seed=key,sample_shape=20)

predictive_mean = predictive_dist.mean()
predictive_std = predictive_dist.stddev()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(x, y, "o", label="Observations", color="tab:red", alpha=0.8, markersize=2.5)
ax.plot(xtest, predictive_mean, label="Predictive mean", color="black")

ax.fill_between(xtest.squeeze(), predictive_mean - predictive_std,
    predictive_mean + predictive_std, alpha=0.2, color="tab:blue", label='Two sigma')
ax.plot(xtest, predictive_mean - predictive_std, color="tab:blue", linestyle="--", linewidth=1)
ax.plot(xtest, predictive_mean + predictive_std, color="tab:blue", linestyle="--", linewidth=1)
ax.plot(xtest, ytest, label="Latent function",color="black", linestyle="--", linewidth=1)


ax.plot(xtest, samples.T, color='tab:blue', alpha=0.8, linewidth=0.2)
[
    ax.axvline(x=z_i, color="black", alpha=0.3, linewidth=1)
    for z_i in learned_params["variational_family"]["inducing_inputs"]
]
ax.legend()
plt.show()

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Daniel Dodd'
