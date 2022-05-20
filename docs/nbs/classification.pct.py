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
# # Classification
#
# In this notebook we demonstrate how to perform inference for Gaussian process models with non-Gaussian likelihoods via maximum a posteriori (MAP) and Markov chain Monte Carlo (MCMC). We focus on a classification task here and use [BlackJax](https://github.com/blackjax-devs/blackjax/) for sampling.

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import blackjax
import gpjax as gpx
import optax as ox

key = jr.PRNGKey(123)

# %% [markdown]
# ## Dataset
#
# With the necessary modules imported, we simulate a dataset $\mathcal{D} = (\boldsymbol{x}, \boldsymbol{y}) = \{(x_i, y_i)\}_{i=1}^{100}$ with inputs $\boldsymbol{x}$ sampled uniformly on $(-1., 1)$ and corresponding binary outputs
#
# $$\boldsymbol{y} = 0.5 * \text{sign}(\cos(2 * \boldsymbol{x} + \boldsymbol{\epsilon})) + 0.5, \quad \boldsymbol{\epsilon} \sim \mathcal{N} \left(\textbf{0}, \textbf{I} * (0.05)^{2} \right).$$
#
# We store our data $\mathcal{D}$ as a GPJax `Dataset` and create test inputs for later.

# %%
x = jnp.sort(jr.uniform(key, shape=(100, 1), minval=-1.0, maxval=1.0), axis=0)
y = 0.5 * jnp.sign(jnp.cos(3 * x + jr.normal(key, shape=x.shape) * 0.05)) + 0.5

D = gpx.Dataset(X=x, y=y)

xtest = jnp.linspace(-1., 1., 500).reshape(-1, 1)
plt.plot(x, y, "o", markersize=8)
# %% [markdown]
# ## MAP inference
#
# We begin by defining a Gaussian process prior with a radial basis function (RBF) kernel, chosen for the purpose of exposition. Since our observations are binary, we choose a Bernoulli likelihood with a probit link function.
# %%
kernel = gpx.RBF()
prior = gpx.Prior(kernel=kernel)
likelihood = gpx.Bernoulli(num_datapoints=D.n)
# %% [markdown]
# We construct the posterior through the product of our prior and likelihood.
# %%
posterior = prior * likelihood
print(type(posterior))
# %% [markdown]
# Whilst the latent function is Gaussian, the posterior distribution is non-Gaussian since our generative model first samples the latent GP and propagates these samples through the likelihood function's inverse link function. This step prevents us from being able to analytically integrate the latent function's values out of our posterior, and we must instead adopt alternative inference techniques. We begin with maximum a posteriori (MAP) estimation, a fast inference procedure to obtain point estimates for the latent function and the kernel's hyperparameters by maximising the marginal log-likelihood. 
# %% [markdown]
# To begin we obtain a set of initial parameter values through the `initialise` callable, and transform these to the unconstrained space via `transform` (see the [regression notebook](https://gpjax.readthedocs.io/en/latest/nbs/regression.html)). We also define the negative marginal log-likelihood, and JIT compile this to accelerate training.
# %%
params, trainable, constrainer, unconstrainer = gpx.initialise(posterior)
params = gpx.transform(params, unconstrainer)

mll = jax.jit(posterior.marginal_log_likelihood(D, constrainer, negative=True))
# %% [markdown]
# We can obtain a MAP estimate by optimising the marginal log-likelihood with Obtax's optimisers.
# %%
opt = ox.adam(learning_rate=0.01)
map_estimate = gpx.fit(
    mll,
    params,
    trainable,
    opt,
    n_iters=500,
)
# %% [markdown]
# However, as a point estimate, MAP estimation is severely limited for uncertainty quantification, providing only a single piece of information about the posterior. On the other hand, through approximate sampling, MCMC methods allow us to learn all information about the posterior distribution.
# %% [markdown]
# ## MCMC inference
#
# At the high level, an MCMC sampler works by starting at an initial position and drawing a sample from a cheap-to-simulate distribution known as the _proposal_. The next step is to determine whether this sample could be considered a draw from the posterior. We accomplish this using an _acceptance probability_ determined via the sampler's _transition kernel_ which depends on the current position and the unnormalised target posterior distribution. If the new sample is more _likely_, we accept it; otherwise, we reject it and stay in our current position. Repeating these steps results in a Markov chain (a random sequence that depends only on the last state) whose stationary distribution (the long-run empirical distribution of the states visited) is the posterior. For a gentle introduction, see the first chapter of [A Handbook of Markov Chain Monte Carlo](https://www.mcmchandbook.net/HandbookChapter1.pdf).
#
# ### MCMC through BlackJax
#
# Rather than implementing a suite of MCMC samplers, GPJax relies on MCMC-specific libraries for sampling functionality. We focus on [BlackJax](https://github.com/blackjax-devs/blackjax/) in this notebook, which we recommend adopting for general applications. However, we also support TensorFlow Probability as demonstrated in the [TensorFlow Probability Integration notebook](https://gpjax.readthedocs.io/en/latest/nbs/tfp_integration.html).
#
# We'll use the No U-Turn Sampler (NUTS) implementation given in BlackJax for sampling. For the interested reader, NUTS is a Hamiltonian Monte Carlo sampling scheme where the number of leapfrog integration steps is computed at each step of the change according to the NUTS algorithm. In general, samplers constructed under this framework are very efficient.
#
# We begin by generating _sensible_ initial positions for our sampler before defining an inference loop and sampling 500 values from our Markov chain. In practice, drawing more samples will be necessary.
# %%
# Adapted from BlackJax's introduction notebook.
num_adapt = 1000
num_samples = 1000

mll = jax.jit(posterior.marginal_log_likelihood(D, constrainer, negative=False))

adapt = blackjax.window_adaptation(
    blackjax.nuts, mll, num_adapt, target_acceptance_rate=0.65
)

# Initialise the chain
last_state, kernel, _ = adapt.run(key, params)

def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos


# Sample from the posterior distribution
states, infos = inference_loop(key, kernel, last_state, num_samples)

# %% [markdown]
# ### Sampler efficiency
#
# BlackJax gives us easy access to our sampler's efficiency through metrics such as the sampler's _acceptance probability_ (the number of times that our chain accepted a proposed sample, divided by the total number of steps run by the chain). For NUTS and Hamiltonian Monte Carlo sampling, we typically seek an acceptance rate of 60-70% to strike the right balance between having a chain which is _stuck_ and rarely moves versus a chain that is too jumpy with frequent small steps.
# %%
acceptance_rate = jnp.mean(infos.acceptance_probability)
print(f"Acceptance rate: {acceptance_rate:.2f}")

# %% [markdown]
# Our acceptance rate is slightly too large, prompting an examination of the chain's trace plots. A well-mixing chain will have very few (if any) flat spots in its trace plot whilst also not having too many steps in the same direction. In addition to the model's hyperparameters, there will be 500 samples for each of the 100 latent function values in the `states.position` dictionary. We depict the chains that correspond to the model hyperparameters and the first value of the latent function for brevity.
# %%
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(15, 5), tight_layout=True)
ax0.plot(states.position['kernel']['lengthscale'])
ax1.plot(states.position['kernel']['variance'])
ax2.plot(states.position['latent'][:, 0, :])
ax0.set_title("Kernel Lengthscale")
ax1.set_title("Kernel Variance")
ax2.set_title("Latent Function (index = 1)")

# %% [markdown]
# ## Prediction
#
# Having obtained samples from the posterior, we draw ten instances from our model's predictive distribution per MCMC sample. Using these draws, we will be able to compute credible values and expected values under our posterior distribution. 
#
# An ideal Markov chain would have samples completely uncorrelated with their neighbours after a single lag. However, in practice, correlations often exist within our chain's sample set. A commonly used technique to try and reduce this correlation is _thinning_ whereby we select every $n$th sample where $n$ is the minimum lag length at which we believe the samples are uncorrelated. Although further analysis of the chain's autocorrelation is required to find appropriate thinning factors, we employ a thin factor of 10 for demonstration purposes.
# %%
thin_factor = 10
samples = []

for i in range(0, num_samples, thin_factor):
    ps = gpx.parameters.copy_dict_structure(params)
    ps['kernel']['lengthscale'] = states.position['kernel']['lengthscale'][i]
    ps['kernel']['variance'] = states.position['kernel']['variance'][i]
    ps['latent'] = states.position['latent'][i, :, :]
    ps = gpx.transform(ps, constrainer)

    predictive_dist = likelihood(posterior(D, ps)(xtest), ps)
    samples.append(predictive_dist.sample(seed=key, sample_shape=(10,)))

samples = jnp.vstack(samples)

lower_ci, upper_ci = jnp.percentile(samples, jnp.array([2.5, 97.5]), axis=0)
expected_val = jnp.mean(samples, axis=0)

# %% [markdown]
#
# Finally, we end this tutorial by plotting the predictions obtained from our model against the observed data.

# %%
fig, ax = plt.subplots(figsize=(16, 5), tight_layout=True)
ax.plot(x, y, "o", markersize=5, color='tab:red', label='Observations', zorder=2, alpha=0.7)
ax.plot(xtest, expected_val, linewidth=2, color='tab:blue', label='Predicted mean', zorder=1)
ax.fill_between(xtest.flatten(), lower_ci.flatten(), upper_ci.flatten(), alpha=0.2, color='tab:blue', label='95% CI')

# %% [markdown]
# ## System configuration

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -a "Thomas Pinder (edited by Daniel Dodd)"
