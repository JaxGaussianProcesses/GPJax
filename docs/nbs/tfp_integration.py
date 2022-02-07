# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: 'Python 3.9.7 64-bit (''gpjax'': conda)'
#     language: python
#     name: python3
# ---

# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import optimizers
import matplotlib.pyplot as plt
from pprint import PrettyPrinter
pp=PrettyPrinter(indent=4)

key = jr.PRNGKey(123)

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
posterior = gpx.Prior(kernel=gpx.RBF()) * gpx.Gaussian(num_datapoints=training.n)

# %%
params, constrainers, unconstrainers = gpx.initialise(posterior)

# %%
import typing as tp

def dict_array_coercion(params) -> tp.Tuple[tp.Callable, tp.Callable]:
    flattened_pytree = jax.tree_util.tree_flatten(params)
    def dict_to_array(parameter_dict) -> jnp.DeviceArray:
        return jax.tree_util.tree_flatten(parameter_dict)
    
    def array_to_dict(parameter_array) -> tp.Dict:
        return jax.tree_util.tree_unflatten(flattened_pytree[1], parameter_array)
    
    return dict_to_array, array_to_dict


# %%
dict_to_array, array_to_dict = dict_array_coercion(params)

# %%
parray = dict_to_array(params)
print(parray)

# %%
array_to_dict(parray)

# %%
param_array = jax.tree_util.tree_flatten(params)[0]
params_reconstruct = jax.tree_util.tree_unflatten(jax.tree_util.tree_flatten(params)[1], param_array)
assert params_reconstruct == params

# %%
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

priors = gpx.parameters.get_param_template(params)
priors['kernel']['lengthscale'] = tfd.Gamma(concentration=jnp.array(1.), rate=jnp.array(1.))
priors['kernel']['variance'] = tfd.Gamma(concentration=jnp.array(1.), rate=jnp.array(1.))
priors['likelihood']['obs_noise'] = tfd.Gamma(concentration=jnp.array(1.), rate=jnp.array(1.))
priors

# %%
mll = posterior.marginal_log_likelihood(training, constrainers,priors=priors, negative=False)
mll(params)

# %%
from functools import partial 


def build_log_pi(params, target):
    def mapper(param_array, params):
        return jax.tree_util.tree_unflatten(jax.tree_util.tree_flatten(params)[1], param_array)

    mapper_fn = partial(mapper, params=params)
    
    def array_mll(parameter_array):
        parameter_dict = mapper_fn([jnp.array(i) for i in parameter_array])
        return target(parameter_dict)
    return array_mll

mll_array_form = build_log_pi(params, mll)

# %%
jax.tree_util.tree_unflatten(jax.tree_util.tree_flatten(params)[1], param_array)

# %%
jax.grad(mll_array_form)(jax.tree_util.tree_flatten(params)[0])

# %%
mll_array_form(jnp.array(jax.tree_util.tree_flatten(params)[0]))


# %%
def run_chain(key, state):
    kernel = tfp.mcmc.NoUTurnSampler(mll_array_form, 1e-1)
    return tfp.mcmc.sample_chain(
        2000,
        current_state=state,
        kernel=kernel,
        trace_fn=lambda _, results: results.target_log_prob,
        seed=key,
    )


# %%
states, log_probs = jax.jit(run_chain)(key, jnp.array(jax.tree_util.tree_flatten(params)[0]))

# %%
states.shape

# %%
plt.plot(states[500:, 1, :], alpha=0.5)

# %%
