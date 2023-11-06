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

# %% [markdown]
# Let's now fit the model. Using the abstraction provided by `GPJaxRegressor`, this can be achieved by simply invoking the `fit` method.

# %%
model.fit(x, y, key=key)

# %% [markdown]
# In the above cell, the Adam optimiser was used as the the default optimiser. However, should you wish to use another optimiser from [Optax](https://github.com/google-deepmind/optax), then you can simply supply it to fit as follows
# ```
# import optax as ox
#
# model.fit(x, y, key=key, optim = ox.sgd(learning_rate=0.01))
# ```

# %% [markdown]
# ## Model evaluation
#
# With a model now fit, we wish to evaluate the model's quality. To do this, we can _score_ the model using either a GPJax scoring function, or a [metric from sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html). We demonstrate how this can be achieved using GPJax's log-posterior density, and the mean-squared error function from sklearn. Scoring functions can be evaluation on either the training or test set, as we show below.

# %%
from sklearn.metrics import mean_squared_error

model.score(xtest, ytest, gpx.sklearn.SKLearnScore("mse", mean_squared_error))
model.score(x, y, gpx.sklearn.LogPredictiveDensity())

# %% [markdown]
# ## Model prediction
#
# Once we're happy with the model's performance, we can make predictions using the `predict` method. Here we are presented with four options:
# 1. `predict` returns the predictive posterior distribution
# 2. `predict_mean` returns the expected value of the predictive posterior distribution
# 3. `predict_stddev` returns the standard-deviation of the predictive posterior distribution
# 4. `predict_mean_and_stddev` returns 2. and 3.

# %%
mu, sigma = model.predict_mean_and_stddev(xtest)

fig, ax = plt.subplots(figsize=(7.5, 2.5))
ax.plot(x, y, "x", label="Observations", color=cols[0], alpha=0.5)
ax.fill_between(
    xtest.squeeze(),
    mu - 2 * sigma,
    mu + 2 * sigma,
    alpha=0.2,
    label="Two sigma",
    color=cols[1],
)
ax.plot(
    xtest,
    mu - 2 * sigma,
    linestyle="--",
    linewidth=1,
    color=cols[1],
)
ax.plot(
    xtest,
    mu + 2 * sigma,
    linestyle="--",
    linewidth=1,
    color=cols[1],
)
ax.plot(
    xtest, ytest, label="Latent function", color=cols[0], linestyle="--", linewidth=2
)
ax.plot(xtest, mu, label="Predictive mean", color=cols[1])
ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))


# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'

# %% [markdown]
#
