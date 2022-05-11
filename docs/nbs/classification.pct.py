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
# In this notebook we show how inference can be done using Markov Chain Monte-Carlo for Gaussian process models with a non-Gaussian likelihood function. We focus on a classification task here and use [BlackJax](https://github.com/blackjax-devs/blackjax/) for sampling.

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import blackjax
import gpjax as gpx

key = jr.PRNGKey(123)

# %% [markdown]
# ## Datasets
#
# We'll simulate a binary dataset where our inputs $x$ are sampled according to $x_i \sim \mathcal{U}(-1., 1.)$ for $1 \leq i \leq 100$. Our corresponding outputs will be calculated according to
# $$ y_i = 0.5*\operatorname{sign}(\cos(2*x + \epsilon_i) + 0.5, $$
# where $\epsilon_i \sim \mathcal{N}(0, 0.01)$. Note, the multiplication and addition of 0.5 is simply to ensure that our outputs are in $\{0, 1\}$.

# %%
x = jnp.sort(jr.uniform(key, shape=(100, 1), minval=-1.0, maxval=1.0), axis=0)
y = 0.5 * jnp.sign(jnp.cos(3 * x + jr.normal(key, shape=x.shape) * 0.05)) + 0.5
xtest = jnp.linspace(-1., 1., 500).reshape(-1, 1)
plt.plot(x, y, "o", markersize=8)

# %%
training = gpx.Dataset(X=x, y=y)

# %% [markdown]
# We can now define our prior Gaussian process such that an RBF kernel has been selected for the purpose of exposition. However, an alternative kernel may be a better choice.

# %%
kern = gpx.RBF()
prior = gpx.Prior(kernel=kern)

# %% [markdown]
# Now we can proceed to define our likelihood function. In this example, our observations are binary, so we will select a Bernoulli likelihood. Using this likelihood function, we can compute the posterior through the product of our likelihood and prior.

# %%
likelihood = gpx.Bernoulli(num_datapoints=training.n)
posterior = prior * likelihood
print(type(posterior))

# %%
params, training_status, constrainer, unconstrainer = gpx.initialise(posterior)
params = gpx.transform(params, unconstrainer)

# %% [markdown]
# With a posterior in place, we can estimate the maximum a posteriori using ObJax's optimisers. However, our Gaussian process is no longer conjugate, meaning that in addition to the kernel's hyperparameters, we are also tasked with learning the values of process' latent function.

# %%
mll = jax.jit(posterior.marginal_log_likelihood(training, constrainer, negative=False))

# %% [markdown]
# ## Markov Chain Monte-Carlo Inference
#
# Whilst the latent function is Gaussian, the posterior distribution is now non-Gaussian. This is because our generative model first samples from the latent function, then propogates these samples through the likelihood function's inverse link function. In general, this step prevents us from being able to analytically integrate the latent function's values out of our posterior, and we must instead use inference technicques such as Markov Chain Monte-Carlo (MCMC). 
#
# At a very high level, MCMC works by first sampling from the posterior distribution before calculating an accpetance probability according to the sampler's transition kernel. If the new sample is more _likely_ according to the target posterior distribution, then we accept the sample, otherwise we reject the sample and stay in our current position. For a gentle introduction, see the first chapter of [A  Handbook of Markov Chain Monte Carlo](https://www.mcmchandbook.net/HandbookChapter1.pdf).
#
# ### MCMC through BlackJax
#
# Rather than implementing a suite of MCMC samplers, GPJax relies on MCMC-specific libraries for sampling functionality. In this notebook we'll focus on [BlackJax](https://github.com/blackjax-devs/blackjax/) and would, in general, advise using BlackJax for MCMC sampling. However, we also support TensorFlow Probability for sampling, as demonstrated in the [TensorFlow Probability Integration notebook](https://gpjax.readthedocs.io/en/latest/nbs/tfp_integration.html).
#
# We'll use the No U-Turn Sampler (NUTS) implementation given in BlackJax for sampling. For the interested reader, NUTS is a Hamiltonian Monte-Carlo sampling scheme where the number of leapfrog integration steps is computed at each step of the change according to the NUTS algorithm. In general, samplers constructed under this framework are very efficient.
#
# In the following cell, we'll first generate _sensible_ initial positions for our sampler before defining an inference loop and sampling 500 values from our Markov chain. In general, drawing more samples will be necessary. The following cell is adapted from BlackJax's introduction notebook.

# %%
num_adapt = 1000
num_samples = 1000

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
# BlackJax gives us easy access to our sampler's efficiency through metrics such as the sampler's acceptance probability. Simply put, this the number of times that our chain accepted a proposed sample, divided by the total number of steps run by the chain. For NUTS and Hamiltonian Monte-Carlo sampling, we typically seek an acceptance rate of 60-70% as this strikes the right balance between having a chain which is _stuck_ and rarely moves, versus a chain that is too jumpy and makes too frequent small steps.

# %%
acceptance_rate = jnp.mean(infos.acceptance_probability)
print(f"Acceptance rate: {acceptance_rate:.2f}")

# %% [markdown]
# In this example, we see that the acceptance rate is slightly too large, so a useful next step would be to inspect the trace plots of the chain's samples. A well mixing chain will have very few, if any, flat spots in its trace plot whilst also not having too many steps in the same direction. In addition to the model's hyperparameters, there will be 500 samples for each of the 100 latent function values in the `states.position` dictionary. For brevity, we will just plot the chains that correspond to the model hyperparameters and the first latent function's value.

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
# With posterior samples now obtained, we'll draw 10 samples from our model's predictive distribution per MCMC sample. Using these draws, we will be able to compute credible values and expected values under our posterior distribution. 
#
# In an ideal Markov chain, each sample would be completely uncorrelated with its neighbouring samples. However, in practice this is not the case and correlations exist within our chain's sample set. A commonly used technique to try and reduce this correlation is called thinning whereby we simply select every $n$-th sample where $n$ is a lag-length at which we believe our samples will be uncorrelated. For demonstratory purposes, I'll use a thin factor of 10 here. In practice, further investigation of the chain's autocorrelation will be necessary to determine optimal thin factors.

# %%
thin_factor = 10
samples = []

for i in range(0, num_samples, thin_factor):
    ps = gpx.parameters.copy_dict_structure(params)
    ps['kernel']['lengthscale'] = states.position['kernel']['lengthscale'][i]
    ps['kernel']['variance'] = states.position['kernel']['variance'][i]
    ps['latent'] = states.position['latent'][i, :, :]
    ps = gpx.transform(ps, constrainer)

    predictive_dist = likelihood(posterior(training, ps)(xtest), ps)
    samples.append(predictive_dist.sample(seed=key, sample_shape=(10,)))

samples = jnp.vstack(samples)

lower_ci, upper_ci = jnp.percentile(samples, jnp.array([2.5, 97.5]), axis=0)
expected_val = jnp.mean(samples, axis=0)

# %% [markdown]
# ### Plotting
#
# We can now plot the predictions obtained from our model and assess its performance relative to the observed dataset that we fit to.

# %%
fig, ax = plt.subplots(figsize=(16, 5), tight_layout=True)
ax.plot(x, y, "o", markersize=5, color='tab:red', label='Observations', zorder=2, alpha=0.7)
ax.plot(xtest, expected_val, linewidth=2, color='tab:blue', label='Predicted mean', zorder=1)
ax.fill_between(xtest.flatten(), lower_ci.flatten(), upper_ci.flatten(), alpha=0.2, color='tab:blue', label='95% CI')

# %% [markdown]
# ## System Configuration

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -a "Thomas Pinder"
