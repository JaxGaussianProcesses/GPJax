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

# %% [markdown]
# # Sparse Regression
#
# In this notebook we demonstrate how to implement sparse variational Gaussian processes (SVGPs) of <strong data-cite="hensman2013gaussian"></strong>. When seeking to model more than ~5000 data points or/and the assumed likelihood is non-Gaussian, SVGPs are a tractable option. However, for models of less than 5000 data points and a Gaussian likelihood function, we recommend using the marginal log-likelihood approach presented in the [regression notebook](https://gpjax.readthedocs.io/en/latest/nbs/regression.html).

# %%
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import tensorflow as tf
from jax import jit
# from jax.example_libraries import optimizers
import optax as ox

import gpjax as gpx

key = jr.PRNGKey(123)

# %% [markdown]
# ## Dataset
#
# With the necessary modules imported, we simulate a dataset $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y}) = \{(x_i, y_i)\}_{i=1}^{5000}$ with inputs $\boldsymbol{x}$ sampled uniformly on $(-5, 5)$ and corresponding binary outputs
#
# $$\boldsymbol{y} \sim \mathcal{N} \left(\sin(4 * \boldsymbol{x}) + \sin(2 * \boldsymbol{x}), \textbf{I} * (0.2)^{2} \right).$$
#
# We store our data $\mathcal{D}$ as a GPJax `Dataset` and create test inputs for later.

# %%
N = 5000
noise = 0.2

x = jr.uniform(key=key, minval=-5.0, maxval=5.0, shape=(N,)).sort().reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-5.5, 5.5, 500).reshape(-1, 1)

# %% [markdown]
# ## Inducing inputs
#
# Despite the many elegant theoretical properties of GPs, their use is greatly hindered by analytic and computational intractabilities. In particular, GPs are burdened with cubic and quadratic costs for inference and memory requirements respectively in the number of data points $n$, making them prohibitive for large data sets. Low rank approximations via sparse GPs offer tractability through augmenting the model with a set of $m$ inducing inputs $\boldsymbol{z} = (z_1, \dotsc, z_m)$ that lie in the same space as $\boldsymbol{x}$. At a high level, these act as a pseudo-dataset, enabling low-rank approximations $\mathbf{K}_{\boldsymbol{z}\boldsymbol{z}}$ of the true covariance matrix $\mathbf{K}_{\boldsymbol{x}\boldsymbol{x}}$ to be computed at lower costs (Quinonero Candela and Rasmussen, 2005). Taking a variational approach, SVGPs reduce these costs further via mini-batching (Hensman et al., 2013) and address non-conjugacy (Hensman et al., 2015). A cost comparison is shown below where $b$ is the mini-batch size:
#
# |    | GPs | sparse GPs | SVGP |
# | -- | -- | -- | -- | 
# | Inference cost | $\mathcal{O}(n^3)$ | $\mathcal{O}(n m^2)$ | $\mathcal{O}(b m^2)$  | 
# | Memory cost    | $\mathcal{O}(n^2)$ | $\mathcal{O}(n m)$ | $\mathcal{O}(n m)$ |


# Initialisation of the inducing inputs is an important area of active research since this affects the algorithm's convergence. For simplicity, we consider a linear spaced grid of 50 points across our observed data's support.

# %%
Z = jnp.linspace(-5.0, 5.0, 50).reshape(-1, 1)

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", alpha=0.3)
ax.plot(xtest, f(xtest))
ax.scatter(Z, jnp.zeros_like(Z), marker="|", color="black")
[ax.axvline(x=z, color="black", alpha=0.3, linewidth=1) for z in Z]
plt.show()

# %% [markdown]
# ## Defining the variational process
#
# Unlike regular GP regression, we do not acess the marginal log-likelihood of our true process. Instead, we introduce a variational approximation $q(f(\cdot))$ that is itself a Gaussian process. We then seek to minimise the Kullback-Leibler divergence $\operatorname{KL}(\cdot || \cdot)$ from our approximate process $q(f(\cdot))$ to the true process $p(f(\cdot)|\mathcal{D})$.

# %%
likelihood = gpx.Gaussian(num_datapoints=N)
p = gpx.Prior(kernel=gpx.RBF()) * likelihood
q = gpx.VariationalGaussian(inducing_inputs=Z)
# %% [markdown]
# We collect our true and approximate posterior Gaussian processes up into an `SVGP` object. This object is simply there to define the variational strategy that we will adopt in the forthcoming inference.

# %%
svgp = gpx.SVGP(posterior=p, variational_family=q)

# %% [markdown]
# ## Inference
#
# ### Evidence lower bound
#
# With a model now defined, we will seek to infer the optimal model hyperparameters $\theta$ and the variational mean $\mathbf{m}$ and covariance $\mathbf{S}$ that define our approximate posterior. To achieve this, we maximise the evidence lower bound (ELBO) with respect to $\{\theta, \mathbf{m}, \mathbf{S} \}$. This is a task that is equivalent to minimising the Kullback-Leibler divergence from the approximate posterior to the true posterior, up to a normalisation constant. For more details on this, see Sections 3.1 and 4.1 of the excellent review paper <strong data-cite="leibfried2020tutorial"></strong>.
#
# As we wish to maximise the ELBO, we'll return it's negative so that minimisation of the negative is equivalent to maximisation of the true ELBO.
#

# %%
params, trainables, constrainers, unconstrainers = gpx.initialise(svgp)
params = gpx.transform(params, unconstrainers)

loss_fn = jit(svgp.elbo(D, constrainers, negative=True))

# %% [markdown]
# ### Mini-batching
#
# Despite introducing a set of inducing points into our model, inference can still be intractable when the observed dataset's size is very large. To circumvent this, optimisation can be done using stochastic mini-batches. The `Dataset` object given in GPJax can easily be batched using the `batch()` method. Further accelerations can be given using prefetching and cacheing in a manner similar to [TensorFlow's Dataset object](https://www.tensorflow.org/guide/data_performance).

# %%
Dbatched = D.cache().repeat().shuffle(D.n).batch(batch_size=128).prefetch(buffer_size=1)

# %%
optimiser = ox.adam(learning_rate=0.01)

learned_params = gpx.abstractions.fit_batches(
    objective = loss_fn,
    train_data = Dbatched, 
    params = params,
    trainables = trainables,
    optax_optim = optimiser,
    n_iters=2500,
)
learned_params = gpx.transform(learned_params, constrainers)

# %% [markdown]
# ## Predictions
#
# With optimisation complete, we are free to use our inferred parameter set to make predictions on a test set of data. This can be achieve in an identical manner to all other GP models within GPJax (see e.g., the [regression notebook](https://gpjax.readthedocs.io/en/latest/nbs/regression.html)).

# %%
latent_dist = svgp(learned_params)(xtest)
predictive_dist = likelihood(latent_dist, learned_params)

# %%
meanf = predictive_dist.mean()
sigma = predictive_dist.stddev()

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", alpha=0.15, label="Training Data", color="tab:gray")
ax.plot(xtest, meanf, label="Posterior mean", color="tab:blue")
ax.fill_between(xtest.flatten(), meanf - sigma, meanf + sigma, alpha=0.3)
[
    ax.axvline(x=z, color="black", alpha=0.3, linewidth=1)
    for z in learned_params["variational_family"]["inducing_inputs"]
]
plt.show()

# %% [markdown]
# ## System information

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder & Daniel Dodd'
