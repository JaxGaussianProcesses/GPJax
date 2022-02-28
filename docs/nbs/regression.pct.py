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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Regression
# %%

from pprint import PrettyPrinter
import gpjax as gpx

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax.experimental import optimizers

pp = PrettyPrinter(indent=4)
key = jr.PRNGKey(123)

# %%
import optax as ox

type(ox.adam(0.01))

# %% [markdown]
# ## Synthetic datasets
#
# With the necessary libraries and modules imported we can simulate some data. We'll sample our inputs $x$ according to $x_1, \ldots , x_{50} \sim \mathcal{U}(-3., 3)$ and corresponding outputs will be,
#
# $$y_i = \sin(4x_i) + \cos(2 x_i) + \epsilon_i,$$
#
# such that $\epsilon_i \sim \mathcal{N}(0, 0.2)$.

# %%
N = 50
noise = 0.2

x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(N,)).sort().reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise
xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
ytest = f(xtest)

# %%
training = gpx.Dataset(X=x, y=y)

# %% [markdown]
# To better understand the data that we have simulated, we can plot both the underlying latent function and the observed data which is subject to Gaussian noise.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xtest, ytest, label="Latent function")
ax.plot(x, y, "o", label="Observations")
ax.legend(loc="best")

# %% [markdown]
# ## Defining the GP prior
#
# We can now define model. A zero-mean Gaussian process places a prior over $f$, such that $p(f) \sim \mathcal{N}(0, K_{xx})$ where $K_{xx}$ is the Gram matrix generated by a user-specified kernel function $k$ such that $[K_{xx}]_{i, j} = k(x_i, x_j)$. The choice of kernel function is an important choice as, among other things, it governs the smoothness of the outputs that our Gaussian process can generate. For now, we'll use a squared exponential kernel which takes the form
#
# $$k(x, x') = \sigma^2 \exp\left(-\frac{\lVert x-x' \rVert_2^2}{2 \ell^2}\right).$$
#
# As we can see, the squared exponential kernel has two parameters; a lengthscale $\ell$ that controls horizontal displacement, and a variance term $\sigma$ that controls vertical displacement.

# %%
kernel = gpx.RBF()
prior = gpx.Prior(kernel=kernel)

# %% [markdown]
# ## Computing the posterior
#
# The posterior distribution is proportional to the prior multiplied by a likelihood function. For this example we'll assume that the likelihood function is a Gaussian distribution. Using this, we can easily compute the posterior using the `*` operator.

# %%
lik = gpx.Gaussian(num_datapoints=training.n)
posterior = prior * lik

# %% [markdown]
# ### Stating parameters
#
# To store our model's hyperparameters, we suggest using a dictionary. In this example, our kernel is parameterised by a lengthscale and variance parameter. Further, our likelihood functions controls the amount of observation noise; the final parameter that we must optimise. These three quantities should therefore be defined as

# %%
params, constrainer, unconstrainer = gpx.initialise(posterior)
pp.pprint(params)

# %% [markdown]
# ### Parameter transformation
#
# The three parameters we defined earlier are all strictly positive. To ensure more stable optimisation, it is strongly advised to transform the parameters onto an unconstrained space first. Through the `transform` function and a given bijective transformation function this can be achieved as follows

# %%
params = gpx.transform(params, unconstrainer)

# %% [markdown]
# ## Optimisation
#
# To learn the optimal set of hyperparameters, we'll optimise the marginal log-likelihood of the posterior with respect to the hyperparameters. To realise this function, one can run the following.

# %%
from jax import jit

mll = jit(posterior.marginal_log_likelihood(training, constrainer, negative=True))
mll(params)

# %% [markdown]
# Note that most optimisers (including here) minimise a given function. Truly, we wish to maximise the marginal log-likelihood, so we instead realise the negative marginal log-likelihood so that the minimisation is equivalent to maximising the marginal log-likelihood.

# %% [markdown]
# ### Defining an optimiser
#
# We can now define an optimiser using one of the optimiser's supplied in Jax's `experimental` module. For this example we'll use the adam optimiser with a step-size of $0.01$.

# %%
opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)

type(opt_init)

# %%
from gpjax.abstractions import fit

opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
final_params = fit(mll, params, opt_init, opt_update, get_params, n_iters=500)

# %% [markdown]
# ## Learned parameters
#
# The exact value of our learned parameters can often be useful in answering certain questions about the underlying process. To find these values, we should untransfom them so that they are redefined on the original constrained space.

# %%
final_params = gpx.transform(final_params, constrainer)
pp.pprint(final_params)

# %% [markdown]
# ## Prediction
#
# ### Direct prediction
#
# Equipped with a posterior distribution and a set of optimised hyperparameter values defined on their original parameter space, we are now in a position to query our GP's predictive posterior distribution at a set of test points. To do this, we can either compute the process' expectation and variance directly using the following `mean` and `variance` functions.

# %%
mu = posterior.mean(training, final_params)(xtest)
sigma = posterior.variance(training, final_params)(xtest)
one_stddev = jnp.sqrt(jnp.diag(sigma))

# %% [markdown]
# With the predictive mean and variance acquired, we can now visualise how well our GP does at explaining the original data.

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", label="Obs", color="tab:red")
ax.plot(xtest, mu, label="pred", color="tab:blue")
ax.fill_between(
    xtest.squeeze(),
    mu.squeeze() - one_stddev,
    mu.squeeze() + one_stddev,
    alpha=0.2,
    color="tab:blue",
)
ax.plot(xtest, mu.squeeze() - one_stddev, color="tab:blue", linestyle="--", linewidth=1)
ax.plot(xtest, mu.squeeze() + one_stddev, color="tab:blue", linestyle="--", linewidth=1)

ax.legend()

# %% [markdown]
# ## System information

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
