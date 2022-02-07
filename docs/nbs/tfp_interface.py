# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: GPJax
#     language: python
#     name: gpjax
# ---

# %% [markdown] colab={"base_uri": "https://localhost:8080/"} id="mfIyurrKVHro" outputId="0232144a-9ba5-4738-d6dd-8c7beb095266" pycharm={"name": "#%% md\n"}
# # TensorFlow Probability Interface

# %%
# %load_ext lab_black

# %% colab={"base_uri": "https://localhost:8080/"} id="RJvA91AKYj52" outputId="88048f85-742e-4163-8eda-4632bcc998f4"
import gpjax
import gpjax.core as gpx
import gpviz as gpv
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp
import matplotlib.pyplot as plt
from jax import grad, jit

tfd = tfp.distributions
key = jr.PRNGKey(123)
plt.style.use(gpv.__stylesheet__)

# %% [markdown] id="qfU3MNxZkR6W"
# ## Simulate some data

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="KBB9v752Yzps" outputId="5e16957a-1a27-4d41-8fb8-3423151e6612"
x = jnp.sort(jr.uniform(key, minval=-5.0, maxval=5.0, shape=(100, 1)), axis=0)
f = lambda x: jnp.sin(jnp.pi * x) / (jnp.pi * x)
y = f(x) + jr.normal(key, shape=x.shape) * 0.05
plt.plot(x, f(x), label="Latent fn")
plt.plot(x, y, "o", label="Observations", alpha=0.6)
plt.legend(loc="best")

# %%
training = gpx.Dataset(X=x, y=y)

# %% [markdown] id="DUAIbmAmkUGn"
# ## Define GP

# %% colab={"base_uri": "https://localhost:8080/"} id="RZLbImDSZBWB" outputId="10be6892-cd65-46d7-dee4-ab5ba1604ae8"
prior = gpx.Prior(kernel=gpx.RBF())
posterior = prior * gpx.Gaussian()
print(posterior)

# %% [markdown] id="j9aKKVP1kcIy"
# ## Initialise parameters
#
# All parameters in this model are constrained to be positive. The softplus transformation will be applied to them to map them onto the entire real line.

# %% colab={"base_uri": "https://localhost:8080/"} id="J4f2GzgfjzVT" outputId="1487a58c-02c6-44d6-8bc5-77caabc7abbe"
params = gpx.initialise(posterior)
configs = gpx.get_defaults()
constrainer, unconstrainer = gpx.build_all_transforms(params.keys(), configs)
params = unconstrainer(params)

# %% [markdown] id="oD1L76w2kzuv"
# ## State priors
#
# We'll also place prior distributions on the constrained value of each of our three parameters.

# %% id="3elDDcZ7kAK4"
priors = {
    "lengthscale": tfd.Gamma(1.0, 1.0),
    "variance": tfd.Gamma(2.0, 2.0),
    "obs_noise": tfd.Gamma(2.0, 2.0),
}

# %% [markdown] id="ScWZ0SmhkW2y"
# ## Define target distribution
#
# The marginal log-likelihood distribution can now be computed from the above GP posterior.

# %% id="EDVEDFgLZmiJ"
mll = gpx.marginal_ll(posterior, transform=constrainer, negative=False)

# %% [markdown] id="3Pre2MSQlAFV"
# ### Check the MLL

# %% colab={"base_uri": "https://localhost:8080/"} id="OY0vJRsukEjr" outputId="8feef5c5-d01d-4cff-a872-dc9c7706bd20"
mll(params, training, priors)

# %% colab={"base_uri": "https://localhost:8080/"} id="y2Q7M_uFkL3j" outputId="746f6eeb-b6ac-4a26-cfec-8f1b93ee8db7"
grad(mll)(params, training, priors)

# %% [markdown] id="rzPxSuJSlDFI"
# ## Setup TFP HMC sampler
#
# To allow our MLL function to accept a Jax array as input, not a set of parameters, we'll define a quick helper function to make an arbitary array to our parameter dictionary.

# %% id="jmebSg83kPWp"
from typing import List


def array_to_dict(varray: jnp.DeviceArray, keys: List):
    pdict = {}
    for val, key in zip(varray, keys):
        pdict[key] = val
    return pdict


# %% [markdown] id="FGn6dDyFlW74"
# We'll also fix all our marginal log-likelihood's arguments except the parameters input.

# %% id="3WK3_VrllURp"
from functools import partial

target_dist = partial(mll, training=training, priors=priors)


# %% id="4ldlTlMQljxi"
def build_log_pi(params, target_fn):
    param_keys = list(params.keys())

    def target(params: jnp.DeviceArray):
        coerced_params = array_to_dict(params, param_keys)
        return target_fn(coerced_params)

    return target


# %% id="ix03mQbmltEO"
log_pi = build_log_pi(params, target_dist)

# %% [markdown] id="mBIa7yMwl16f"
# So now we have a functional representation of our GP posterior's marginal log-likelihood that accepts a single Jax array as input. It's worthwhile here checking that evaluation of our new function agrees with our initial marginal log-likelihood function.

# %% colab={"base_uri": "https://localhost:8080/"} id="v7LlBFNbl04z" outputId="7ffe7c78-a1f7-497c-a49e-369ca97b5234"
initial_params = jnp.array(list(params.values())).squeeze()
print(initial_params)

# %% id="VqLGTRVKmO_B"
assert log_pi(initial_params) == mll(params, training, priors)


# %% [markdown] id="3kDRLnUdmsf1"
# With both the marginal log-likelihood and its respective gradient functions in agreeance, we can go ahead and instantiate the TFP HMC sampler. In the interest of time, we'll just sample 500 times from out targer distribution. However, in practice, you'll want to sample for longer.

# %% id="o5b8DfiemZlu"
def run_chain(key, state):
    kernel = tfp.mcmc.NoUTurnSampler(log_pi, 1e-1)
    return tfp.mcmc.sample_chain(
        500,
        current_state=state,
        kernel=kernel,
        trace_fn=lambda _, results: results.target_log_prob,
        seed=key,
    )


# %% colab={"base_uri": "https://localhost:8080/"} id="K70frhaYm7O1" outputId="bc1f4dce-4daa-45b5-8678-ec1292f027f4"
states, log_probs = jit(run_chain)(key, initial_params)

# %% colab={"base_uri": "https://localhost:8080/"} id="pWvImdz5nFE1" outputId="a5c26b9b-f970-4d16-ae70-9f823e80ff99"
states

# %%
constrained_states = jnp.hstack(
    (
        i.reshape(-1, 1)
        for i in constrainer(array_to_dict(states.T, params.keys())).values()
    )
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 350} id="qOU3XXZevhyw" outputId="d768a7da-e574-4e5d-ef9b-811dd8f94812"
burn_in = 100
thin_factor = 2
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))

for i, p, pc, t in zip(range(3), states.T, constrained_states.T, params.keys()):
    ax[0][i].plot(p[burn_in:][::thin_factor])
    ax[0][i].set_title(t.replace("_", " "))
    ax[0][i].set_ylabel("Sampled parameter value (unconstrained)")
    ax[1][i].plot(pc[burn_in:][::thin_factor])
    ax[1][i].set_xlabel("Sample index (post-thinning)")
    ax[1][i].set_ylabel("Sampled parameter value (constrained)")

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -a "Thomas Pinder"
