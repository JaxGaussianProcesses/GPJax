# -*- coding: utf-8 -*-
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
# # Deep Kernel Learning
#
# In this notebook we demonstrate how GPJax can be used in conjunction with
# [Haiku](https://github.com/deepmind/dm-haiku) to build deep kernel Gaussian
# processes. Modelling data with discontinuities is a challenging task for regular
# Gaussian process models. However, as shown in
# <strong data-cite="wilson2016deep"></strong>, transforming the inputs to our
# Gaussian process model's kernel through a neural network can offer a solution to this.

# %%
import typing as tp
from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax.config import config
from jaxtyping import Array, Float
from scipy.signal import sawtooth
from flax import linen as nn 
from simple_pytree import static_field

import gpjax as gpx
import gpjax.kernels as jk
from gpjax.kernels import DenseKernelComputation
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.base import param_field

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
key = jr.PRNGKey(123)

# %% [markdown]
# ## Dataset
#
# As previously mentioned, deep kernels are particularly useful when the data has
# discontinuities. To highlight this, we will use a sawtooth function as our data.

# %%
n = 500
noise = 0.2

key, subkey = jr.split(key)
x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n,)).reshape(-1, 1)
f = lambda x: jnp.asarray(sawtooth(2 * jnp.pi * x))
signal = f(x)
y = signal + jr.normal(subkey, shape=signal.shape) * noise

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-2.0, 2.0, 500).reshape(-1, 1)
ytest = f(xtest)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(x, y, "o", label="Training data", alpha=0.5)
ax.plot(xtest, ytest, label="True function")
ax.legend(loc="best")

# %% [markdown]
# ## Deep kernels
#
# ### Details
#
# Instead of applying a kernel $k(\cdot, \cdot')$ directly on some data, we seek to
# apply a _feature map_ $\phi(\cdot)$ that projects the data to learn more meaningful
# representations beforehand. In deep kernel learning, $\phi$ is a neural network
# whose parameters are learned jointly with the GP model's hyperparameters. The
# corresponding kernel is then computed by $k(\phi(\cdot), \phi(\cdot'))$. Here
# $k(\cdot,\cdot')$ is referred to as the _base kernel_.
#
# ### Implementation
#
# Although deep kernels are not currently supported natively in GPJax, defining one is
# straightforward as we now demonstrate. Using the base `AbstractKernel` object given
# in GPJax, we provide a mixin class named `_DeepKernelFunction` to facilitate the
# user supplying the neural network and base kernel of their choice. Kernel matrices
# are then computed using the regular `gram` and `cross_covariance` functions.


# %%
import flax
from dataclasses import field
from typing import Any
from simple_pytree import static_field
    

@dataclass
class DeepKernelFunction(AbstractKernel):
    base_kernel: AbstractKernel = None
    network: nn.Module = static_field(None)
    dummy_x: jax.Array = static_field(None)
    key: jr.PRNGKeyArray = static_field(jr.PRNGKey(123))
    nn_params: Any = field(init=False, repr=False)
    
    def __post_init__(self):

        if self.base_kernel is None:
            raise ValueError("base_kernel must be specified")
        
        if self.network is None:
            raise ValueError("network must be specified")


        self.nn_params = flax.core.unfreeze(self.network.init(key, self.dummy_x))


    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "1"]:
        state = self.network.init(self.key, x)
        xt = self.network.apply(state, x)
        yt = self.network.apply(state, y)
        return self.base_kernel(xt, yt)


# %% [markdown]
# ### Defining a network
#
# With a deep kernel object created, we proceed to define a neural network. Here we
# consider a small multi-layer perceptron with two linear hidden layers and ReLU
# activation functions between the layers. The first hidden layer contains 32 units,
# while the second layer contains 64 units. Finally, we'll make the output of our
# network a single unit. However, it would be possible to project our data into a
# $d-$dimensional space for $d>1$. In these instances, making the
# [base kernel ARD](https://gpjax.readthedocs.io/en/latest/nbs/kernels.html#Active-dimensions)
# would be sensible.
# Users may wish to design more intricate network structures for more complex tasks,
# which functionality is supported well in Haiku.


# %%
class Network(nn.Module):
  """A simple MLP."""
  @nn.compact
  def __call__(self, x):
      x = nn.Dense(features=4)(x)
      x = nn.relu(x)
      x = nn.Dense(features=2)(x)
      x = nn.relu(x)
      x = nn.Dense(features=1)(x)
      return x
  

forward_linear = Network()
state = jax.jit(forward_linear.init)(key, jnp.ones(x.shape[-1]))

# %% [markdown]
# ## Defining a model
#
# Having characterised the feature extraction network, we move to define a Gaussian
# process parameterised by this deep kernel. We consider a third-order Matérn base
# kernel and assume a Gaussian likelihood. Parameters, trainability status and
# transformations are initialised in the usual manner.

# %%
base_kernel = gpx.RBF()
kernel = DeepKernelFunction(network=forward_linear, base_kernel=base_kernel, key=key, dummy_x=x)
meanf = gpx.Zero()
prior = gpx.Prior(mean_function=meanf, kernel=kernel)
likelihood = gpx.Gaussian(num_datapoints=D.n)
posterior = prior * likelihood
# %% [markdown]
# ### Optimisation
#
# We train our model via maximum likelihood estimation of the marginal log-likelihood.
# The parameters of our neural network are learned jointly with the model's
# hyperparameter set.
#
# With the inclusion of a neural network, we take this opportunity to highlight the
# additional benefits gleaned from using
# [Optax](https://optax.readthedocs.io/en/latest/) for optimisation. In particular, we
# showcase the ability to use a learning rate scheduler that decays the optimiser's
# learning rate throughout the inference. We decrease the learning rate according to a
# half-cosine curve over 1000 iterations, providing us with large step sizes early in
# the optimisation procedure before approaching more conservative values, ensuring we
# do not step too far. We also consider a linear warmup, where the learning rate is
# increased from 0 to 1 over 50 steps to get a reasonable initial learning rate value.

# %%
negative_mll = gpx.ConjugateMLL(negative=True)

# %%
schedule = ox.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=0.01,
    warmup_steps=50,
    decay_steps=1_000,
    end_value=0.0,
)

optimiser = ox.chain(
    ox.clip(1.0),
    ox.adamw(learning_rate=schedule),
)

opt_posterior, history = gpx.fit(
    model=posterior,
    objective=gpx.ConjugateMLL(negative=True),
    train_data=D,
    optim=optimiser,
    num_iters=2500,
)

# %% [markdown]
# ## Prediction
#
# With a set of learned parameters, the only remaining task is to predict the output
# of the model. We can do this by simply applying the model to a test data set.

# %%
latent_dist = posterior(learned_params, D)(xtest)
predictive_dist = likelihood(learned_params, latent_dist)

predictive_mean = predictive_dist.mean()
predictive_std = predictive_dist.stddev()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", label="Observations", color="tab:red")
ax.plot(xtest, predictive_mean, label="Predictive mean", color="tab:blue")
ax.fill_between(
    xtest.squeeze(),
    predictive_mean - predictive_std,
    predictive_mean + predictive_std,
    alpha=0.2,
    color="tab:blue",
    label="One sigma",
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
ax.legend()

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder (edited by Daniel Dodd)'
