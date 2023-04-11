# %% [markdown]
# # Sparse Gaussian Process Regression
#
# In this notebook we consider sparse Gaussian process regression (SGPR)
# <strong data-cite="titsias2009">Titsias (2009)</strong>. This is a solution for
# medium to large-scale conjugate regression problems.
# In order to arrive at a computationally tractable method, the approximate posterior
# is parameterized via a set of $m$ pseudo-points $\boldsymbol{z}$. Critically, the
# approach leads to $\mathcal{O}(nm^2)$ complexity for approximate maximum likelihood
# learning and $O(m^2)$ per test point for prediction.

# %%
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax import jit
from jax.config import config

import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
key = jr.PRNGKey(123)

# %% [markdown]
# ## Dataset
#
# With the necessary modules imported, we simulate a dataset
# $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y}) = \{(x_i, y_i)\}_{i=1}^{500}$
# with inputs $\boldsymbol{x}$ sampled uniformly on $(-3., 3)$ and corresponding
# independent noisy outputs
#
# $$\boldsymbol{y} \sim \mathcal{N} \left(\sin(7\boldsymbol{x}) + x \cos(2 \boldsymbol{x}), \textbf{I} * 0.5^2 \right).$$
#
# We store our data $\mathcal{D}$ as a GPJax `Dataset` and create test inputs and
# labels for later.

# %%
n = 2500
noise = 0.5

key, subkey = jr.split(key)
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
f = lambda x: jnp.sin(2 * x) + x * jnp.cos(5 * x)
signal = f(x)
y = signal + jr.normal(subkey, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-3.1, 3.1, 500).reshape(-1, 1)
ytest = f(xtest)

# %% [markdown]
# To better understand what we have simulated, we plot both the underlying latent
# function and the observed data that is subject to Gaussian noise. We also plot an
# initial set of inducing points over the space.

# %%
n_inducing = 50
z = jnp.linspace(-3.0, 3.0, n_inducing).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(xtest, ytest, label="Latent function", color="tab:blue")
ax.plot(x, y, "o", color="tab:orange", alpha=0.4, label="Observations", markersize=2.5)
[ax.axvline(x=z_i, color="tab:gray", alpha=0.3, linewidth=1) for z_i in z]
ax.legend(loc="best")
plt.show()

# %% [markdown]
# Next we define the true posterior model for the data - note that whilst we can define
# this, it is intractable to evaluate.

# %%
meanf = gpx.Constant()
kernel = gpx.RBF()
likelihood = gpx.Gaussian(num_datapoints=D.n)
prior = gpx.Prior(mean_function=meanf, kernel=kernel)
posterior = prior * likelihood

# %% [markdown]
# We now define the SGPR model through `CollapsedVariationalGaussian`. Through a
# set of inducing points $\boldsymbol{z}$ this object builds an approximation to the
# true posterior distribution. Consequently, we pass the true posterior and initial
# inducing points into the constructor as arguments.

# %%
q = gpx.CollapsedVariationalGaussian(posterior=posterior, inducing_inputs=z)

# %% [markdown]
# We define our variational inference algorithm through `CollapsedVI`. This defines
# the collapsed variational free energy bound considered in
# <strong data-cite="titsias2009">Titsias (2009)</strong>.

# %%
elbo = jit(gpx.CollapsedELBO(negative=True))

# %% [markdown]
# We now train our model akin to a Gaussian process regression model via the `fit`
# abstraction. Unlike the regression example given in the
# [conjugate regression notebook](https://gpjax.readthedocs.io/en/latest/nbs/regression.html),
# the inducing locations that induce our variational posterior distribution are now
# part of the model's parameters. Using a gradient-based optimiser, we can then
# _optimise_ their location such that the evidence lower bound is maximised.

# %%
opt_posterior, history = gpx.fit(
    model=q,
    objective=elbo,
    train_data=D,
    optim=ox.adamw(learning_rate=1e-2),
    num_iters=500,
)

plt.plot(history)

# %% [markdown]
# We show predictions of our model with the learned inducing points overlayed in grey.

# %%
latent_dist = opt_posterior(xtest, train_data=D)
predictive_dist = opt_posterior.posterior.likelihood(latent_dist)

inducing_points = opt_posterior.inducing_inputs

samples = latent_dist.sample(seed=key, sample_shape=(20,))

predictive_mean = predictive_dist.mean()
predictive_std = predictive_dist.stddev()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(x, y, "o", label="Observations", color="tab:orange", alpha=0.4, markersize=2.5)
ax.plot(xtest, predictive_mean, label="Predictive mean", color="tab:blue")

ax.fill_between(
    xtest.squeeze(),
    predictive_mean - predictive_std,
    predictive_mean + predictive_std,
    alpha=0.2,
    color="tab:blue",
    label="Two sigma",
)
ax.plot(
    xtest,
    predictive_mean - predictive_std,
    color="tab:blue",
    linestyle="--",
    linewidth=1,
)
ax.plot(
    xtest,
    predictive_mean + predictive_std,
    color="tab:blue",
    linestyle="--",
    linewidth=1,
)
ax.plot(
    xtest,
    ytest,
    label="Latent function",
    color="tab:green",
    linestyle="--",
    linewidth=1,
)


ax.plot(xtest, samples.T, color="tab:blue", alpha=0.8, linewidth=0.2)
[ax.axvline(x=z_i, color="tab:gray", alpha=0.3, linewidth=1) for z_i in inducing_points]
ax.legend()
plt.show()

# %% [markdown]
# ## Runtime comparison
#
# Given the size of the data being considered here, inference in a GP with a full-rank
# covariance matrix is possible, albeit quite slow. We can therefore compare the
# speedup that we get from using the above sparse approximation with corresponding
# bound on the marginal log-likelihood against the marginal log-likelihood in the
# full model.

# %%
full_rank_model = gpx.Prior(mean_function=gpx.Zero(), kernel=gpx.RBF()) * gpx.Gaussian(
    num_datapoints=D.n
)
negative_mll = jit(gpx.ConjugateMLL(negative=True))
# %timeit negative_mll(full_rank_model, D).block_until_ready()

# %%
negative_elbo = jit(gpx.CollapsedELBO(negative=True))
# %timeit negative_elbo(q, D).block_until_ready()

# %% [markdown]
# As we can see, the sparse approximation given here is around 50 times faster when
# compared against a full-rank model.

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Daniel Dodd'
