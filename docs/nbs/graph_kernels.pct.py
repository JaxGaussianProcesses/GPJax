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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Graph Kernels
#
# This notebook demonstrates how regression models can be constructed on the vertices of a graph using a Gaussian process. To achieve this, we'll use the Matérn kernel presented in  <strong data-cite="borovitskiy2021matern"></strong>.

# %%
import random

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import networkx as nx
import optax as ox
from jax import jit

import gpjax as gpx

key = jr.PRNGKey(123)

# %% [markdown]
# ## Graph construction
#
# Our graph $\mathcal{G}=\lbrace V, E \rbrace$ is comprised of set of vertices $V = \lbrace v_1, v_2, \ldots, v_n\rbrace$ and edges $E=\lbrace (v_i, v_j)\in V \ \text{such that} \ i \neq j\rbrace$. We will be considering the [Barbell graph](https://en.wikipedia.org/wiki/Barbell_graph) in this notebook that is an undirected graph containing two clusters of vertices with a single shared edge between the two clusters.
#
# Contrary to the typical barbell graph, we'll randomly remove a subset of 30 edges within each of the two clusters. Given the 40 vertices within the graph, this results in 351 edges. We can see this construction in the following cell.

# %%
vertex_per_side = 20
n_edges_to_remove = 30
p = 0.8

G = nx.barbell_graph(vertex_per_side, 0)

random.seed(123)
[G.remove_edge(*i) for i in random.sample(list(G.edges), n_edges_to_remove)]

pos = nx.spring_layout(G, seed=123)  # positions for all nodes

nx.draw(G, pos, node_color="tab:blue", with_labels=False, alpha=0.5)

# %% [markdown]
#
# ### Computing the graph Laplacian
#
# Graph kernels use the _graph Laplacian_ matrix $L$ to quantify the smoothness of a signal, or function, on the graph. To calculate this, we compute $$L=D-A$$ where $D$ is the diagonal _degree matrix_ $D$ that contains the vertices' degree along the diagonal and the _adjacency matrix_ $A$ that has $(i,j)^{\text{th}}$ entry of 1 if $v_i, v_j$ are connected and 0 otherwise. Using `networkx` gives us an easy way to compute this.

# %%
L = nx.laplacian_matrix(G).toarray()

# %% [markdown]
#
# ## Simulating a signal on the graph
#
# Our task is to construct a Gaussian process $f$ that maps from the graph's vertex set $V$ onto the real line $\mathbb{R}$. To that end, we must now simulate a signal on the graph's vertices that we will go on to try and predict. We use a single draw from a Gaussian process prior to simulate our response values $y$ where we hardcode parameter values. The correspodning input value set for this model, typically denoted $x$, is the index set of the graph's vertices.
# %%
xs = jnp.arange(G.number_of_nodes()).reshape(-1, 1)

kernel = gpx.GraphKernel(laplacian=L)
f = gpx.Prior(kernel=kernel)

true_params = f.params
true_params["kernel"] = {
    "lengthscale": jnp.array(2.3),
    "variance": jnp.array(3.2),
    "smoothness": jnp.array(6.1),
}

rv = f(true_params)(xs)
y = rv.sample(seed=key).reshape(-1, 1)

D = gpx.Dataset(X=xs, y=y)

# %% [markdown]
#
# We can visualise this signal in the following cell
# %%
nx.draw(G, pos, node_color=y, with_labels=False, alpha=0.5)

vmin, vmax = y.min(), y.max()
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax)
)
sm.set_array([])
cbar = plt.colorbar(sm)

# %% [markdown]
#
# ## Constructing a graph Gaussian process
#
# With an observed dataset created, we can now proceed to define our posterior Gaussian process and optimise the model's hyperparameters. Whilst our underlying space is now the graph's vertex set and is therefore non-Euclidean, our likelihood is still Gaussian and the model is still conjugate. For this reason, we simply have to perform gradient descent on the GP's marginal log-likelihood term. We do this using the adam optimiser provided in `optax`.

# %%
likelihood = gpx.Gaussian(num_datapoints=y.shape[0])
posterior = f * likelihood
params, training_status, constrainer, unconstrainer = gpx.initialise(posterior)
params = gpx.transform(params, unconstrainer)

mll = jit(
    posterior.marginal_log_likelihood(
        train_data=D, transformations=constrainer, negative=True
    )
)

opt = ox.adam(learning_rate=0.01)
learned_params = gpx.abstractions.optax_fit(
    objective=mll,
    params=params,
    trainables=training_status,
    optax_optim=opt,
    n_iters=1000,
    jit_compile=True,
)
learned_params = gpx.transform(learned_params, constrainer)

# %% [markdown]
#
# ## Making predictions
#
# With an optimised set of parameters, we can now make predictions on the graph. We haven't defined a training and testing dataset here, so we'll simply query the predictive posterior for the full graph, compute the root-mean-squared error (RMSE) for the model using the initialised parameters and the optimise set, and finally compare the RMSE values.
# %%
initial_dist = likelihood(posterior(D, params)(xs), params)
predictive_dist = likelihood(posterior(D, learned_params)(xs), learned_params)

initial_mean = initial_dist.mean()
learned_mean = predictive_dist.mean()

rmse = lambda ytrue, ypred: jnp.sum(jnp.sqrt(jnp.square(ytrue - ypred)))

initial_rmse = jnp.sum(jnp.sqrt(jnp.square(y.squeeze() - initial_mean)))
learned_rmse = jnp.sum(jnp.sqrt(jnp.square(y.squeeze() - learned_mean)))
print(
    f"RMSE with initial parameters: {initial_rmse: .2f}\nRMSE with learned parameters:"
    f" {learned_rmse: .2f}"
)

# %% [markdown]
#
# We can also plot the source of error in our model's predictions on the graph by the following.
# %%
error = jnp.abs(learned_mean - y.squeeze())

nx.draw(G, pos, node_color=error, with_labels=False, alpha=0.5)

vmin, vmax = error.min(), error.max()
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax)
)
sm.set_array([])
cbar = plt.colorbar(sm)
# %% [markdown]
#
# Reassuringly, our model seems to be giving equally good predictions in each cluster.

# %% [markdown]
# ## System Configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
