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
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox 

key = jr.PRNGKey(123)

# %% [markdown]
# # Gaussian Process Barycentres
#
#

# %%
n_data = 100
n_datasets = 5

x = jnp.linspace(-5., 5., n_data).reshape(-1, 1)
xtest = jnp.linspace(-5.5, 5.5, 500).reshape(-1, 1)
f = lambda x, a, b: a+jnp.sin(b*x) 

ys = []
for i in range(n_datasets):
    key, subkey = jr.split(key)
    vertical_shift = jr.uniform(subkey, minval=0., maxval=2.)
    period = jr.uniform(subkey, minval=0.75, maxval=1.25)
    noise_amount = jr.uniform(subkey, minval=0.01, maxval=0.5)
    noise = jr.normal(subkey, shape = x.shape) * noise_amount
    ys.append(f(x, vertical_shift, period) + noise)

y = jnp.hstack(ys)

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(x, y, 'o')


# %%
def fit_gp(x, y):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    D = gpx.Dataset(X = x, y=y)
    posterior = gpx.Prior(kernel=gpx.RBF()) * gpx.Gaussian(num_datapoints= n_data)
    params, trainables, constrainers, unconstrainers = gpx.initialise(posterior)
    params = gpx.transform(params, unconstrainers)

    objective = jax.jit(posterior.marginal_log_likelihood(D, constrainers, negative=True))

    opt = ox.adam(learning_rate=0.01)
    learned_params = gpx.optax_fit(objective=objective, trainables=trainables, params=params, optax_optim=opt, n_iters=100, jit_compile=True, log_rate=None)
    learned_params = gpx.transform(learned_params, constrainers)

    mu = posterior.mean(D, learned_params)(xtest)
    cov = posterior.variance(D, learned_params)(xtest)
    return mu, cov


# %%
out = jax.vmap(fit_gp, in_axes=(None, 1), out_axes=1)(x, y)

# %%
plt.plot(xtest, out[0].squeeze())

# %%
