# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Barycentre of Gaussian Processes
#
# In this notebook we'll give an implementation of <strong data-cite="mallasto2017learning"></strong>. In this work, the existence of a Wasserstein barycentre between a collection of Gaussian processes is proven.

# %%
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
import distrax as dx
import typing as tp
import jax.scipy.linalg as jsl

key = jr.PRNGKey(123)

# %% [markdown]
# ## Background
#
# ### Wasserstein distance
#
# The 2-Wasserstein distance metric between two probability measures $\mu$ and $\nu$ quantifies the minimal cost required to transport the unit mass from $\mu$ to $\nu$, or vice-versa. Typically, computing this metric requires the solution of a linear program. However, when $\mu$ and $\nu$ both belong to the family of multivariate Gaussian distributions, the solution is analytically given by
# $$W_2^2(\mu, \nu) = \lVert m_1, m_2 \rVert^2_2 + \operatorname{Tr}(S_1 + S_2 - 2(S_1^{1/2}S_2S_1^{1/2})^{1/2})\tag{1}$$
# where $\mu \sim \mathcal{N}(m_1, S_1)$ and $\nu\sim\mathcal{N}(m_2, S_2)$.
#
# ### Wasserstein barycentre
#
# For a collection of $T$ measures $\lbrace\mu_i\rbrace_{t=1}^T \in \mathcal{P}_2(\theta)$, the Wasserstein barycentre $\bar{\mu}$ is the measure that minimises the average Wasserstein distance to all other measures in the set. More formally, the Wasserstein barycentre is the Fréchet mean on a Wasserstein space that we can write as
# $$\bar{\mu} = \argmin_{\mu\in\mathcal{P}_2(\theta)}\sum_{t=1}^T \alpha_t W_2^2(\mu, \mu_t)\tag{2}$$
# where $\alpha\in\mathbb{R}^T$ is a weight vector that sums to 1.
#
# As with the Wasserstein distance, identifying the Wasserstein barycentre $\bar{\mu}$ is often an computationally demanding optimisation problem. However, when all the measures admit a multivariate Gaussian density, the barycentre $\bar{\mu} = \mathcal{N}(\bar{m}, \bar{S})$ has analytical solutions
# $$\bar{m} = \sum_{t=1}^T \alpha_t m_t\,, \quad \bar{S}=\sum_{t=1}^T\alpha_t (\bar{S}^{1/2}S_t\bar{S}^{1/2})^{1/2}\,.\tag{3}$$
# Identifying $\bar{S}$ is achieved through a fixed-point iterative update.
#
# ## Barycentre of Gaussian processes
#
# It was shown in <strong data-cite="mallasto2017learning"></strong> that the barycentre $\bar{f}$ of a collection of Gaussian processes $\lbrace f_i\rbrace_{i=1}^T$ such that $f_i \sim \mathcal{GP}(m, K)$ can be found using the same solutions as in (3). In this notebook, we will demonstrate how this can be achieved in GPJax.
#
# ## Data
#
# We'll simulate five datasets over which we'll first learn a Gaussian process posterior, and then identify the Gaussian process barycentre at a set of test points. Each dataset will be a $\sin$ function, each with differing vertical shift, periodicity and noise amounts

# %%
n_data = 100
n_test = 200
n_datasets = 5

x = jnp.linspace(-5.0, 5.0, n_data).reshape(-1, 1)
xtest = jnp.linspace(-5.5, 5.5, n_test).reshape(-1, 1)
f = lambda x, a, b: a + jnp.sin(b * x)

ys = []
for i in range(n_datasets):
    key, subkey = jr.split(key)
    vertical_shift = jr.uniform(subkey, minval=0.0, maxval=2.0)
    period = jr.uniform(subkey, minval=0.75, maxval=1.25)
    noise_amount = jr.uniform(subkey, minval=0.01, maxval=0.5)
    noise = jr.normal(subkey, shape=x.shape) * noise_amount
    ys.append(f(x, vertical_shift, period) + noise)

y = jnp.hstack(ys)

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(x, y, "o")
plt.show()


# %% [markdown]
# ## Learning a posterior distribution
#
# We'll now learn a posterior distribution over each of the datasets using and independent Gaussian process. We won't spend ant time here discussing how a GP can be optimised or how a kernel can be fit. For advice on achieving this, see the [Regression notebook](https://gpjax.readthedocs.io/en/latest/nbs/regression.html) for advice on optimsation and the [Kernels notebook](https://gpjax.readthedocs.io/en/latest/nbs/kernels.html) for advice on selecting an appropriate kernel.

# %%
def fit_gp(x, y):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    D = gpx.Dataset(X=x, y=y)
    likelihood = gpx.Gaussian(num_datapoints=n_data)
    posterior = gpx.Prior(kernel=gpx.RBF()) * likelihood
    params, trainables, constrainers, unconstrainers = gpx.initialise(posterior)
    params = gpx.transform(params, unconstrainers)

    objective = jax.jit(posterior.marginal_log_likelihood(D, constrainers, negative=True))

    opt = ox.adam(learning_rate=0.01)
    learned_params = gpx.optax_fit(
        objective=objective,
        trainables=trainables,
        params=params,
        optax_optim=opt,
        n_iters=1000,
        jit_compile=True,
        log_rate=None,
    )
    learned_params = gpx.transform(learned_params, constrainers)
    return likelihood(posterior(D, learned_params)(xtest), learned_params)


posterior_preds = [fit_gp(x, i) for i in ys]


# %% [markdown]
# ## Computing the barycentre
#
# In GPJax, the predictive distribution of a GP is given by a [Distrax](https://github.com/deepmind/distrax) distribution. This makes it straightforward to then extract the mean vector and covariance matrix of the GP that will be used for learning a barycentre. In the following cell, we'll implement the fixed point scheme given in (3). We'll make use of Jax's `vmap` operator here and speed up potentially large matrix operations using broadcasting in `tensordot`.

# %%
def sqrtm(A):
    return jnp.real(jsl.sqrtm(A))


def wasserstein_barycentres(distributions: tp.List[dx.Distribution], weights: jnp.DeviceArray):
    covariances = [d.covariance() for d in distributions]
    cov_stack = jnp.stack(covariances)
    stack_sqrt = jax.vmap(sqrtm)(cov_stack)

    def step(covariance_candidate, i):
        inner_term = jax.vmap(sqrtm)(
            jnp.matmul(jnp.matmul(stack_sqrt, covariance_candidate), stack_sqrt)
        )
        fixed_point = jnp.tensordot(weights, inner_term, axes=1)
        return fixed_point, fixed_point

    return step


# %% [markdown]
# With a function defined that'll allow us to learn a Barycentre, we'll now compute it using Jax's `lax.scan` operator. This speeds up for loops in Jax and a nice introduction to this can be found in the [Jax documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html).

# %%
weights = jnp.ones((n_datasets,)) / n_datasets

means = jnp.stack([d.mean() for d in posterior_preds])
barycentre_mean = jnp.tensordot(weights, means, axes=1)

step_fn = jax.jit(wasserstein_barycentres(posterior_preds, weights))
initial_covariance = jnp.eye(n_test)

barycentre_covariance, sequence = jax.lax.scan(step_fn, initial_covariance, jnp.arange(10))


barycentre_process = dx.MultivariateNormalFullCovariance(barycentre_mean, barycentre_covariance)


# %% [markdown]
# ## Plotting the result
#
# With a barycentre learned, we can visualise the result. We can see that the result looks sensible as it follows the sinusoidal curve of all the inferred GPs and the uncertainty bands look sensible.

# %%
def plot(dist, ax, color="tab:blue", label=None, ci_alpha: float = 0.2, linewidth: float = 1.0):
    mu = dist.mean()
    sigma = dist.stddev()
    ax.plot(xtest, dist.mean(), linewidth=linewidth, color=color, label=label)
    ax.fill_between(xtest.squeeze(), mu - sigma, mu + sigma, alpha=ci_alpha, color=color)


fig, ax = plt.subplots(figsize=(16, 5))
[plot(d, ax, color="tab:blue", ci_alpha=0.1) for d in posterior_preds]
plot(barycentre_process, ax, color="tab:red", label="Barycentre", ci_alpha=0.4, linewidth=2)

# %% [markdown]
# ## System Information

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
