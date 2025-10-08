# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: gpjax
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Graph Edge Kernels
#
# This notebook demonstrates how link prediction model can be constructed on the vertices
# of a graph using a Gaussian process kernel like RBF which helps to learn edge wise covariances on the edges
# with an Edge kernel presented in <strong data-cite="(Yu and Chu, 2008)"></strong>.
# For a general discussion of the kernels supported within GPJax, see the
# [kernels notebook](https://docs.jaxgaussianprocesses.com/_examples/constructing_new_kernels).
# %%
import random

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optax as ox
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

import gpjax as gpx
from gpjax.kernels.non_euclidean.graph_edge import GraphEdgeKernel
from gpjax.parameters import (
    Parameter,
)

# %% [markdown]
# ## Graph construction
#
# Our graph $\mathcal{G}=\lbrace V, E \rbrace$ comprises a set of vertices
# $V = \lbrace v_1, v_2, \ldots, v_n\rbrace$ and edges
# $E=\lbrace (v_i, v_j)\in V \ : \ i \neq j\rbrace$. In particular, we will consider
# a [barbell graph](https://en.wikipedia.org/wiki/Barbell_graph) that is an undirected
# graph containing two clusters of vertices with a single shared edge between the
# two clusters.
#
# Contrary to the typical barbell graph, we'll randomly remove a subset of 30 edges
# within each of the two clusters. Given the 40 vertices within the graph, this results
# in 351 edges as shown below.

# %%
key = jr.key(42)

vertex_per_side = 20
n_edges_to_remove = 30
p = 0.8

G = nx.barbell_graph(vertex_per_side, 0)

random.seed(123)
[G.remove_edge(*i) for i in random.sample(list(G.edges), n_edges_to_remove)]

pos = nx.spring_layout(G, seed=123)  # positions for all nodes

nx.draw(G, pos, node_size=100, edge_color="black", with_labels=False)

# ## Simulating a signal on the graph
#
# We begin by simulating a signal on the graph's vertices which will help on edge Gaussian Process.
# For this example its all random feature matrix which goes into the base kernel (RBF in this case),
# a binary indication of edge presence or absense also randomly drawn.
# %%
node_feature_matrix = np.random.uniform(low=0.5, high=13.3, size=(40, 5))

# %%
np_y = np.array([0] * 175 + [1] * (176))
np.random.shuffle(np_y)
y = jnp.array(np_y).reshape(-1, 1).astype(jnp.float64)

# %%
edge_indices = jnp.array(G.edges).astype(jnp.int64)

# %%
D = gpx.Dataset(X=edge_indices, y=y.astype(jnp.float64))

# %% [markdown]
#
# ## Constructing a edge graph Gaussian process
#
# With our dataset created, we proceed to define our posterior Gaussian process and
# optimise the model's hyperparameters.
# Whilst our underlying space is the graph's vertex set and is therefore
# non-Euclidean, our likelihood is still Gaussian and the model is non-conjugate.
# We simply perform gradient descent on the GP's marginal
# log-likelihood term as in the
# [classification notebook](https://docs.jaxgaussianprocesses.com/_examples/classification/).

# %%
kernel = gpx.kernels.RBF()

# %%
graph_kernel = GraphEdgeKernel(
    feature_mat=node_feature_matrix,
    base_kernel=kernel,
)

# %%
meanf = gpx.mean_functions.Constant()
prior = gpx.gps.Prior(mean_function=meanf, kernel=graph_kernel)
likelihood = gpx.likelihoods.Bernoulli(num_datapoints=D.n)

# %%
posterior = prior * likelihood

# %%
optimiser = ox.adam(learning_rate=0.01)
opt_posterior, history = gpx.fit(
    model=posterior,
    objective=lambda p, d: -gpx.objectives.log_posterior_density(p, d),
    train_data=D,
    optim=ox.adamw(learning_rate=0.01),
    num_iters=1000,
    key=key,
    trainable=Parameter,  # train all parameters (default behavior)
)
# %% [markdown]
#
# ## Making predictions
#
# Having optimised our hyperparameters, we can now make predictions on the graph.
# Though we haven't defined a training and testing dataset here, we'll simply query
# the predictive posterior for the full graph to compare the ROC-AUC, F1, accuracy scores.
# %%
initial_dist = likelihood(posterior(edge_indices, D))
predictive_dist = opt_posterior.likelihood(opt_posterior(edge_indices, D))

# %%
predictive_mean = predictive_dist.mean
predictive_std = jnp.sqrt(predictive_dist.variance)

# %%
y_prob = predictive_mean
auc = roc_auc_score(y, y_prob)
accu = accuracy_score(y, jnp.where(y_prob > 0.5, 1, 0))
f1 = f1_score(y, jnp.where(y_prob > 0.5, 1, 0))
print(f"ROC AUC:, {auc} || Accuracy: {accu} || F1: {f1}")

# %%
fpr, tpr, thresholds = roc_curve(y, y_prob)

# %%
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
