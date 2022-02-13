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
#     display_name: Python 3.9.7 ('gpjax')
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Graph Kernels
#
# This notebook demonstrates how regression models can be constructed on the vertices of a graph using a Gaussian process. To achieve this, we'll use the Matérn kernel presented in <cite data-cite="borovitskiy2021matern"/>

# %%
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import matplotlib.pyplot as plt
import random

key = jr.PRNGKey(123)

# %%
vertex_per_side = 20
p = 0.8

G = nx.barbell_graph(vertex_per_side, 0)

random.seed(123)
[G.remove_edge(*i) for i in random.sample(list(G.edges), 30)]
xs = []
ys = []
for idx, (node, subkey) in enumerate(zip(G, jr.split(key, 2 * vertex_per_side))):
    if idx < vertex_per_side:
        y = jr.bernoulli(subkey, p).astype(int)
    else:
        y = jr.bernoulli(subkey, 1.0 - p).astype(int)
    G.nodes[node]["label"] = y
    xs.append(idx)
    ys.append(y)


pos = nx.spring_layout(G, seed=123)  # positions for all nodes
cols = []
for i in ys:
    if i == 0:
        cols.append("tab:blue")
    elif i == 1:
        cols.append("tab:orange")
nx.draw(G, pos, node_color=cols, with_labels=False, alpha=0.5)

# %%
L = nx.laplacian_matrix(G).toarray()
plt.matshow(L)

# %%
kernel = gpx.GraphKernel(laplacian=L)
f = gpx.Prior(kernel=kernel)
true_params = f.params
true_params["kernel"] = {
    "lengthscale": jnp.array(2.3),
    "variance": jnp.array(3.2),
    "smoothness": jnp.array(6.1),
}
rv = f.random_variable(jnp.array(xs).reshape(-1, 1), true_params)

# %%
y = rv.sample(seed=key).reshape(-1, 1)
y.shape
D = gpx.Dataset(X=jnp.array(xs).reshape(-1, 1), y=y)

# %%
posterior = f * gpx.Gaussian(num_datapoints=y.shape[0])
params, constrainer, unconstrainer = gpx.initialise(posterior)
params = gpx.transform(params, unconstrainer)

# %%
from jax import jit

mll = jit(posterior.marginal_log_likelihood(training=D, transformations=constrainer, negative=True))

# %%
mll(params)

# %%
import optax as ox

opt = ox.adam(learning_rate=0.01)
learned_params = gpx.abstractions.optax_fit(
    objective=mll, params=params, optax_optim=opt, n_iters=1000
)
learned_params = gpx.transform(learned_params, constrainer)

# %%
mu = posterior.mean(D, params)(jnp.array(xs).reshape(-1, 1))
jnp.sum(jnp.sqrt(jnp.square(y - mu)))

# %%
mu = posterior.mean(D, learned_params)(jnp.array(xs).reshape(-1, 1))
jnp.sum(jnp.sqrt(jnp.square(y - mu)))
