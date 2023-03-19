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
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Regression
#
# In this notebook we demonstate how to fit a Gaussian process regression model.

# %%
from pprint import PrettyPrinter

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax import jit
from jax.config import config
from jaxutils import Dataset, fit
import jaxkern as jk

import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
pp = PrettyPrinter(indent=4)
key = jr.PRNGKey(123)

# %% [markdown]
# ## Dataset
#
# With the necessary modules imported, we simulate a dataset $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y}) = \{(x_i, y_i)\}_{i=1}^{100}$ with inputs $\boldsymbol{x}$ sampled uniformly on $(-3., 3)$ and corresponding independent noisy outputs
#
# $$\boldsymbol{y} \sim \mathcal{N} \left(\sin(4\boldsymbol{x}) + \cos(2 \boldsymbol{x}), \textbf{I} * 0.3^2 \right).$$
#
# We store our data $\mathcal{D}$ as a GPJax `Dataset` and create test inputs and labels for later.

# %%
n = 100
noise = 0.3

x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).sort().reshape(-1, 1)


def f(x):
    return jnp.sin(4 * x) + jnp.cos(2 * x)


signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise

D = Dataset(X=x, y=y)

xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
ytest = f(xtest)

# %% [markdown]
# To better understand what we have simulated, we plot both the underlying latent function and the observed data that is subject to Gaussian noise.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xtest, ytest, label="Latent function")
ax.plot(x, y, "o", label="Observations")
ax.legend(loc="best")

# %% [markdown]
# Our aim in this tutorial will be to reconstruct the latent function from our noisy observations $\mathcal{D}$ via Gaussian process regression. We begin by defining a Gaussian process prior in the next section.

# %% [markdown]
# ## Defining the prior
#
# A zero-mean Gaussian process (GP) places a prior distribution over real-valued functions $f(\cdot)$ where $f(\boldsymbol{x}) \sim \mathcal{N}(0, \mathbf{K}_{\boldsymbol{x}\boldsymbol{x}})$ for any finite collection of inputs $\boldsymbol{x}$.
# Here $\mathbf{K}_{\boldsymbol{x}\boldsymbol{x}}$ is the Gram matrix generated by a user-specified symmetric, non-negative definite kernel function $k(\cdot, \cdot')$ with $[\mathbf{K}_{\boldsymbol{x}\boldsymbol{x}}]_{i, j} = k(x_i, x_j)$.
# The choice of kernel function is critical as, among other things, it governs the smoothness of the outputs that our GP can generate.
# For simplicity, we consider a radial basis function (RBF) kernel:
# $$k(x, x') = \sigma^2 \exp\left(-\frac{\lVert x - x' \rVert_2^2}{2 \ell^2}\right).$$
#
# On paper a GP is written as $f(\cdot) \sim \mathcal{GP}(\textbf{0}, k(\cdot, \cdot'))$, we can reciprocate this process in GPJax via defining a `Prior` with our chosen `RBF` kernel.

# %%
kernel = jk.RBF()
prior = gpx.Prior(kernel=kernel)

# %% [markdown]
#
# The above construction forms the foundation for GPJax's models. Moreover, the GP prior we have just defined can be represented by a [Distrax](https://github.com/deepmind/distrax) multivariate Gaussian distribution. Such functionality enables trivial sampling, and mean and covariance evaluation of the GP.

# %%
parameter_state = prior.init_params(key)
prior_dist = prior(parameter_state.params)(xtest)

prior_mean = prior_dist.mean()
prior_std = prior_dist.stddev()
samples = prior_dist.sample(seed=key, sample_shape=(20,)).T

plt.plot(xtest, samples, color="tab:blue", alpha=0.5)
plt.plot(xtest, prior_mean, color="tab:orange")
plt.fill_between(
    xtest.flatten(),
    prior_mean - prior_std,
    prior_mean + prior_std,
    color="tab:orange",
    alpha=0.3,
)
plt.show()

# %% [markdown]
# ## Constructing the posterior
#
# Having defined our GP, we proceed to define a description of our data $\mathcal{D}$ conditional on our knowledge of $f(\cdot)$ --- this is exactly the notion of a likelihood function $p(\mathcal{D} | f(\cdot))$. While the choice of likelihood is a critical in Bayesian modelling, for simplicity we consider a Gaussian with noise parameter $\alpha$
# $$p(\mathcal{D} | f(\cdot)) = \mathcal{N}(\boldsymbol{y}; f(\boldsymbol{x}), \textbf{I} \alpha^2).$$
# This is defined in GPJax through calling a `Gaussian` instance.

# %%
likelihood = gpx.Gaussian(num_datapoints=D.n)

# %% [markdown]
# The posterior is proportional to the prior multiplied by the likelihood, written as
#
#   $$ p(f(\cdot) | \mathcal{D}) \propto p(f(\cdot)) * p(\mathcal{D} | f(\cdot)). $$
#
# Mimicking this construct, the posterior is established in GPJax through the `*` operator.

# %%
posterior = prior * likelihood

# %% [markdown]
# <!-- ## Hyperparameter optimisation
#
# Our kernel is parameterised by a length-scale $\ell^2$ and variance parameter $\sigma^2$, while our likelihood controls the observation noise with $\alpha^2$. Using Jax's automatic differentiation module, we can take derivatives of  -->
#
# ## Parameter state
#
# So far, all of the objects that we've defined have been stateless. To give our model state, we can use the `init_params` method in GPJax. Upon calling this, a `Parameters` class is returned that contains four dictionaries:
#
# | Dictionary  | Description  |
# |---|---|
# |  `params` | Initial parameter values.  |
# | `trainable`  | Boolean dictionary that determines the training status of parameters (`True` for being trained and `False` otherwise).  |
# |  `bijectors` | Bijectors that can map parameters between the _unconstrained space_ and their original _constrained space_.  |
# |  `priors` | Where relevant, the prior distribution that is specified for each parameter on its _constrained space_.  |
#

# %%
params = posterior.init_params(key)

# %% [markdown]
# Note, for this example a key is not strictly necessary as none of the parameters are stochastic variables. Howeveror some models, such as the sparse spectrum GP, the parameters are themselves random variables and the key is therefore essential.

# %% [markdown]
# To motivate the purpose the `bijectors` more precisely, notice that our model hyperparameters $\{\ell^2, \sigma^2, \alpha^2 \}$ are all strictly positive, bijectors act to unconstrain these during the optimisation proceedure.

# %% [markdown]
# To train our hyperparameters, we optimising the marginal log-likelihood of the posterior with respect to them. We define the marginal log-likelihood with `marginal_log_likelihood` on the posterior.

# %%
negative_mll = gpx.ConjugateMLL(posterior=posterior, negative=True)
negative_mll(params, D)

# %% [markdown]
# Since most optimisers (including here) minimise a given function, we have realised the negative marginal log-likelihood and just-in-time (JIT) compiled this to accelerate training.

# %% [markdown]
# We can now define an optimiser with `optax`. For this example we'll use the `adamw` optimiser.

# %%
optimiser = ox.adamw(learning_rate=0.01)

learned_params = fit(objective=negative_mll, train_data=D, optim=optimiser, num_iters=500
)

# %% [markdown]
# The returned variable from the `fit` function is another `Parameters` object that contains the learned values.

# %% [markdown]
# ## Prediction
#
# Equipped with the posterior and a set of optimised hyperparameter values, we are now in a position to query our GP's predictive distribution at novel test inputs. To do this, we use our defined `posterior` and `likelihood` at our test inputs to obtain the predictive distribution as a `Distrax` multivariate Gaussian upon which `mean` and `stddev` can be used to extract the predictive mean and standard deviatation.

# %%
latent_dist = posterior(learned_params, D)(xtest)
predictive_dist = likelihood(learned_params, latent_dist)

predictive_mean = predictive_dist.mean()
predictive_std = predictive_dist.stddev()

# %% [markdown]
# With the predictions and their uncertainty acquired, we illustrate the GP's performance at explaining the data $\mathcal{D}$ and recovering the underlying latent function of interest.

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", label="Observations", color="tab:red")
ax.plot(xtest, predictive_mean, label="Predictive mean", color="tab:blue")
ax.fill_between(
    xtest.squeeze(),
    predictive_mean - 2 * predictive_std,
    predictive_mean + 2 * predictive_std,
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
    xtest, ytest, label="Latent function", color="black", linestyle="--", linewidth=1
)

ax.legend()

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder & Daniel Dodd'
