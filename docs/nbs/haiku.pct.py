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
# In this notebook we demonstrate how GPJax can be used in conjunction with [Haiku](https://github.com/deepmind/dm-haiku) to build deep kernel Gaussian processes <strong data-cite="wilson2016deep"></strong>.

# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
import distrax as dx
import typing as tp
import haiku as hk
from gpjax.kernels import Kernel
from chex import dataclass
from scipy.signal import sawtooth

key = jr.PRNGKey(123)

# %% [markdown]
# ## Data
#
# Modelling data with discontinuities is a challenging task for regular Gaussian process models. However, as shown in <strong data-cite="wilson2016deep"></strong>, transforming the inputs to our Gaussian process model's kernel through a neural network can offer a solution to this. To highlight this, we'll model a sawtooth function.

# %%
N = 500
noise = 0.2

x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(N,)).sort().reshape(-1, 1)
f = lambda x: jnp.asarray(sawtooth(2 * jnp.pi * x))
signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise
xtest = jnp.linspace(-2.0, 2.0, 500).reshape(-1, 1)
ytest = f(xtest)

training = gpx.Dataset(X=x, y=y)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(x, y, "o", label="Training data", alpha=0.5)
ax.plot(xtest, ytest, label="True function")
ax.legend(loc="best")

# %% [markdown]
# ## Deep Kernels
#
# ### Details
#
# Typically when we evaluate a kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ on a pair of inputs $x, x' \in \mathcal{X}$, we would compute $k(x, x')$. However, deep kernel regression seeks to apply a transform $\phi: \mathcal{X} \to \mathcal{X}$ to the inputs that seeks to project the inputs into a more meaningful representation. In deep kernel learning, $\phi$ is a neural network whose parameters are learned jointly with the GP model's hyperparameters. The corresponding kernel then computed by $k(\phi(x), \phi(x'))$.
#
# ### Implementation
#
# Deep kernels are not natively supported in GPJax right now. However, defining one is a straightforward task that we demonstrate in the following cell. Using the base `Kernel` object given in GPJax, we'll provide a mixin class named `_DeepKernelFunction` that requires the user to supply a neural network and base kernel. The neural network is responsible for transforming the inputs which will then be consumed by the base kernel. Kernel matrices are then computed using the regular `gram` and `cross_covariance` functions.

# %%
@dataclass
class _DeepKernelFunction:
    network: hk.Module
    base_kernel: Kernel


@dataclass
class DeepKernelFunction(Kernel, _DeepKernelFunction):
    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> jnp.ndarray:
        xt = self.network.apply(params=params, x=x)
        yt = self.network.apply(params=params, x=y)
        return self.base_kernel(xt, yt, params=params)

    def initialise(self, dummy_x, key):
        nn_params = self.network.init(rng=key, x=dummy_x)
        base_kernel_params = self.base_kernel.params
        self._params = {**nn_params, **base_kernel_params}

    @property
    def params(self):
        return self._params


# %% [markdown]
# ### Defining a network
#
# With a deep kernel object now defined, we can define a network that will transform the inputs. For this notebook, we'll use a small multi-layer perceptron with two linear hidden layers and relu activation functions between the layes. The first hidden layer contains 32 units and the second layer contains 64 units. As we are doing 1D regression here, the final output layer of the network is a single unit.
#
# For more complex tasks, users may wish to define more complex network achitectures, the functionality for which is well supported in Haiku.

# %%
def forward(x):
    mlp = hk.Sequential(
        [
            hk.Linear(32),
            jax.nn.relu,
            hk.Linear(64),
            jax.nn.relu,
            hk.Linear(1),
        ]
    )
    return mlp(x)


forward_linear1 = hk.transform(forward)
forward_linear1 = hk.without_apply_rng(forward_linear1)

# %% [markdown]
# ## Defining a model
#
# Now we have defined the feature extraction network that is to be used within our deep kernel, we can now define a Gaussian process that is parameterised by this kernel. We'll use a third-order Matérn kernel as our base kernel and assume a Gaussian likelihood function. Parameters, trainability status and transformations are initialised in the usual manner.

# %%
base_kernel = gpx.Matern52()
kernel = DeepKernelFunction(network=forward_linear1, base_kernel=base_kernel)
kernel.initialise(x, key)
prior = gpx.Prior(kernel=kernel)
likelihood = gpx.Gaussian(num_datapoints=training.n)
posterior = prior * likelihood

params, training_status, constrainers, unconstrainers = gpx.initialise(posterior)
params = gpx.transform(params, unconstrainers)

# %% [markdown]
# ### Optimisation
#
# We train our model using maximum likelihood estimation of the marginal log-likelihood. The parameters of our neural network are learned jointly with the model's hyperparameter set.
#
# With the inclusion of a neural network, we take this opportunity to highlight the additional benefits that can be gleaned from using [Optax](https://optax.readthedocs.io/en/latest/) for optimisation. In particular, we showcase here the ability to use a learning rate scheduler that decays the optimiser's learning rate throughout the inference. In the following cell, the learning rate is decayed over 1000 iterations according to a half-cosine curve. This provides us with large step-sizes early on in the optimisation procedure before decreasing the learning rate to a more conservative value that ensures we do not step too far. A linear warmup is also used which simply means that over 50 steps we increase the learning rate from 0. to 1. to find a sensible intial learning rate value.
# %%
mll = jax.jit(posterior.marginal_log_likelihood(training, constrainers, negative=True))
mll(params)

schedule = ox.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1.0,
    warmup_steps=50,
    decay_steps=1_000,
    end_value=0.0,
)

opt = ox.chain(
    ox.clip(1.0),
    ox.adamw(learning_rate=schedule),
)

final_params = gpx.abstractions.optax_fit(
    mll,
    params,
    training_status,
    opt,
    n_iters=5000,
)
final_params = gpx.transform(final_params, constrainers)

# %% [markdown]
# ## Prediction
#
# With a set of learned parameters, the only remaining task is to predict the output of the model. We can do this by simply applying the model to a test data set.

# %%
latent_dist = posterior(training, final_params)(xtest)
predictive_dist = likelihood(latent_dist, final_params)

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
ax.plot(xtest, predictive_mean - predictive_std, color="tab:blue", linestyle="--", linewidth=1)
ax.plot(xtest, predictive_mean + predictive_std, color="tab:blue", linestyle="--", linewidth=1)
ax.legend()

# %% [markdown]
# ## System information

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
