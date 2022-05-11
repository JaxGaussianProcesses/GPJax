# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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


# %%
def sqrtm(A):
    return jnp.real(jsl.sqrtm(A))

def wasserstein_distance(alpha: dx.MultivariateNormalFullCovariance, beta: dx.MultivariateNormalFullCovariance):
    m0 = alpha.mean()
    m1 = beta.mean()
    K0 = alpha.covariance() + jnp.eye(n_test)*1e-8
    K1 = beta.covariance() + jnp.eye(n_test)*1e-8
    return jnp.linalg.norm(m0 - m1, ord=2) + jnp.sum(jnp.diag(K0 + K1)) - 2*jnp.sum(jnp.diag(sqrtm(sqrtm(K0)@K1@sqrtm(K0))))


def wasserstein_barycentres(distributions: tp.List[dx.Distribution], weights: jnp.DeviceArray):
    covariances = [d.covariance() for d in distributions]
    cov_stack = jnp.stack(covariances)
    stack_sqrt = jax.vmap(sqrtm)(cov_stack)
    
    def step(covariance_candidate, i):
        inner_term = jax.vmap(sqrtm)(jnp.matmul(jnp.matmul(stack_sqrt, covariance_candidate), stack_sqrt))
        fixed_point = jnp.tensordot(weights, inner_term, axes=1)
        return fixed_point, fixed_point

    return step

def fit_gp(x, y):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    D = gpx.Dataset(X = x, y=y)
    likelihood = gpx.Gaussian(num_datapoints= n_data)
    posterior = gpx.Prior(kernel=gpx.RBF()) * likelihood
    params, trainables, constrainers, unconstrainers = gpx.initialise(posterior)
    params = gpx.transform(params, unconstrainers)

    objective = jax.jit(posterior.marginal_log_likelihood(D, constrainers, negative=True))

    opt = ox.adam(learning_rate=0.01)
    learned_params = gpx.optax_fit(objective=objective, trainables=trainables, params=params, optax_optim=opt, n_iters=1000, jit_compile=True, log_rate=None)
    learned_params = gpx.transform(learned_params, constrainers)
    return likelihood(posterior(D, learned_params)(xtest), learned_params)

def plot(dist, ax, color="tab:blue", label = None):
    mu = dist.mean()
    sigma = dist.stddev()
    ax.plot(xtest, dist.mean(), linewidth=1, color=color, label=label)
    ax.fill_between(xtest.squeeze(), mu-sigma, mu+sigma, alpha=0.2, color=color)



# %% [markdown]
# # Gaussian Process Barycentres
#
#

# %%
key = jr.PRNGKey(123)
n_data = 100
n_test = 200

# f1 = lambda x: x**2
# f2 = lambda x: 5.+ -0.15*x**4

f1 = lambda x: jnp.sin(4*x) + 2.5*jnp.cos(5*x)
f2 = lambda x: jnp.sin(2*x) + jnp.cos(6*x)

x1 = jr.uniform(key, minval=-3., maxval=3., shape=(n_data, 1))
key, subkey = jr.split(key)
y1 = f1(x1) + jr.normal(key, shape=(n_data, 1))*0.2
key, subkey = jr.split(subkey)
x2 = jr.uniform(subkey, minval=-3., maxval=3., shape=(n_data, 1))
key, subkey = jr.split(subkey)
y2 = f2(x2) + jr.normal(subkey, shape=(n_data, 1))*0.35
xtest = jnp.linspace(-3., 3., n_test).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(x1, y1, 'o', color='tab:orange')
ax.plot(x2, y2, 'o', color='tab:blue')


# %%
p1 = fit_gp(x1, y1)
p2 = fit_gp(x2, y2)

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(x1, y1, 'o', color='tab:orange')
ax.plot(x2, y2, 'o', color='tab:blue')
plot(p1, ax, color="tab:orange")
plot(p2, ax, color="tab:blue")

# %%
weight_idxs = [0.5] + jnp.linspace(0.5, 1., 10).tolist() + jnp.repeat(1., 5).tolist() + jnp.linspace(1., 0., 20).tolist()+ jnp.repeat(0., 5).tolist()+ jnp.linspace(0., 0.5, 10).tolist()
for idx, i in enumerate(weight_idxs):
    weights = jnp.array([i, 1-i])
    step_fn = jax.jit(wasserstein_barycentres([p1, p2], weights))
    initial_covariance = jnp.eye(n_test)

    barycentre_covariance, sequence = jax.lax.scan(step_fn, initial_covariance, jnp.arange(10))

    means = jnp.stack([d.mean() for d in [p1, p2]])
    barycentre_mean = jnp.tensordot(weights, means, axes=1)
    barycentre_process = dx.MultivariateNormalFullCovariance(barycentre_mean, barycentre_covariance)

    fig, ax = plt.subplots(figsize=(16, 5), tight_layout=True)
    plot(p1, ax, color="tab:green", label=r"$\mu_1$")
    plot(p2, ax, color="tab:blue", label=r"$\mu_2$")
    plot(barycentre_process, ax, color="tab:red", label=r"$\bar{\mu}$")
    ax.legend(loc='lower center',fancybox=True,prop=dict(size=16))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    
    plt.savefig(f'docs/nbs/barycentre_figs/{str(idx).zfill(2)}.png')
    plt.close()

# %%
n_datasets = 5

x = jnp.linspace(-5., 5., n_data).reshape(-1, 1)
xtest = jnp.linspace(-5.5, 5.5, n_test).reshape(-1, 1)
f = lambda x, a, b: a+jnp.sin(b*x) 

ys = []
for i in range(n_datasets):
    key, subkey = jr.split(key)
    vertical_shift = jr.uniform(subkey, minval=0., maxval=2.)
    period = jr.uniform(subkey, minval=0.75, maxval=1.25)
    noise_amount = jr.uniform(subkey, minval=0.01, maxval=0.5)
    noise = jr.normal(subkey, shape = x.shape) * noise_amount
    ys.append(f(x, vertical_shift, period) + noise)

y = jnp.hstack(ys)

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(x, y, 'o')

# %%

# %%
posterior_preds = [fit_gp(x, i) for i in ys]

# %%
weights = jnp.ones((n_datasets,)) / n_datasets
step_fn = jax.jit(wasserstein_barycentres(posterior_preds, weights))
initial_covariance = jnp.eye(n_test)

barycentre_covariance, sequence = jax.lax.scan(step_fn, initial_covariance, jnp.arange(10))

means = jnp.stack([d.mean() for d in posterior_preds])
barycentre_mean = jnp.tensordot(weights, means, axes=1)

# %%
barycentre_process = dx.MultivariateNormalFullCovariance(barycentre_mean, barycentre_covariance)

# %%
fig, ax = plt.subplots(figsize=(16, 5))
[plot(d, ax) for d in posterior_preds]
ax.plot(xtest, barycentre_process.mean(), linewidth=2, color='tab:red')
ax.fill_between(xtest.squeeze(), barycentre_process.mean()-barycentre_process.stddev(), barycentre_process.mean()+barycentre_process.stddev(), alpha=0.2, color='tab:red')


# %%
