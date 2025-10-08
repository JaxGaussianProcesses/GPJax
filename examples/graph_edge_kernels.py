# %%
import jax.numpy as jnp
import jax
from gpjax.parameters import (
    Parameter,
)

import networkx as nx
import gpjax as gpx
import jax.random as jr

from gpjax.kernels.non_euclidean.graph_edge import GraphEdgeKernel
from sklearn.model_selection import train_test_split

import random
import numpy as np
import optax as ox

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve

# %%
key = jr.key(42)

vertex_per_side = 20
n_edges_to_remove = 30
p = 0.8

G = nx.barbell_graph(vertex_per_side, 0)

random.seed(123)
[G.remove_edge(*i) for i in random.sample(list(G.edges), n_edges_to_remove)]

pos = nx.spring_layout(G, seed=123)  # positions for all nodes

nx.draw(
    G, pos, node_size=100, edge_color="black", with_labels=False
)

# %%
attrib_matrix = nx.attr_matrix(G)
L = nx.laplacian_matrix(G).toarray()

# %%
x = np.random.uniform(low=0.5, high=13.3, size=(40, 5))

# %%
np_y = np.array([0] * 175 + [1] * (176))
np.random.shuffle(np_y)
y = jnp.array(np_y).reshape(-1, 1).astype(jnp.float64)

# %%
edge_indices = jnp.array(G.edges)

# %%
edge_indices.shape

# %%
D = gpx.Dataset(X=edge_indices, y=y.astype(jnp.float64))

# %%
kernel = gpx.kernels.RBF()

# %%
graph_kernel = GraphEdgeKernel(
    feature_mat=x,
    base_kernel=kernel,
)

# %%
meanf = gpx.mean_functions.Constant()
prior = gpx.gps.Prior(mean_function=meanf, kernel=graph_kernel)
likelihood = gpx.likelihoods.Bernoulli(num_datapoints=D.n)

# %%
posterior = prior * likelihood

# %%
x = graph_kernel.gram(edge_indices).to_dense()

# %%
x.shape

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

# %%
map_latent_dist = opt_posterior.predict(D.X, train_data=D)

# %%
predictive_dist = opt_posterior.likelihood(map_latent_dist)

# %%
predictive_mean = predictive_dist.mean
predictive_std = jnp.sqrt(predictive_dist.variance)

# %%
predictive_dist.mean

# %%



