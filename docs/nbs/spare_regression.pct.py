# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
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

# %%
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import jit
from jax.experimental import optimizers

key = jr.PRNGKey(123)

# %%
N = 5000
noise = 0.2

x = jr.uniform(key=key, minval=-5.0, maxval=5.0, shape=(N,)).sort().reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise
xtest = jnp.linspace(-5.0, 5.0, 500).reshape(-1, 1)

# %%
Z = jnp.linspace(-5.0, 5.0, 50).reshape(-1, 1)

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", alpha=0.3)
ax.plot(xtest, f(xtest))
ax.scatter(Z, jnp.zeros_like(Z), marker="|", color="black")
[ax.axvline(x=z, color="black", alpha=0.3, linewidth=1) for z in Z]
plt.show()

# %%
D = gpx.Dataset(X=x, y=y)
true_process = gpx.Prior(kernel=gpx.RBF()) * gpx.Gaussian(num_datapoints=N)

# %%
q = gpx.VariationalGaussian(inducing_inputs=Z)

# %%
svgp = gpx.SVGP(true_process, q)

# %%
params, trainables, constrainers, unconstrainers = gpx.initialise(svgp)
params = gpx.transform(params, unconstrainers)

loss_fn = jit(gpx.VFE(svgp, D, constrainers))

# %%
opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)

batched_dataset = jit(gpx.abstractions.mini_batcher(D, batch_size=64, prefetch_buffer=1))

learned_params = gpx.abstractions.fit_batches(
    loss_fn,
    params,
    trainables,
    opt_init,
    opt_update,
    get_params,
    get_batch=batched_dataset,
    n_iters=750,
)
learned_params = gpx.transform(learned_params, constrainers)

# %%
meanf = svgp.mean(learned_params)(xtest)
# varfs = svgp.variance(learned_params)(xtest)

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", alpha=0.3, label="Training Data", color="tab:gray")
ax.plot(xtest, meanf, label="Posterior mean", color="tab:blue")

# %%
