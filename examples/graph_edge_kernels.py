# %%
import jax.numpy as jnp
import gpjax
from gpjax.parameters import (
    Parameter,
)

import networkx as nx
import gpjax as gpx
import jax.random as jr

from gpjax.kernels.non_euclidean.graph_edge_kernel import GraphEdgeKernel
from sklearn.model_selection import train_test_split

import random
import numpy as np
import optax as ox

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
node_feat_matrix = nx.attr_matrix(G)

# %%
x = node_feat_matrix[0]

# %%
np_y = np.array([0] * 20 + [1] * (20))
np.random.shuffle(np_y)
y = jnp.array(np_y).reshape(-1, 1).astype(jnp.float64)

# %%
D = gpx.Dataset(X=x, y=y.astype(jnp.float64))

# %%
kernel = gpx.kernels.RBF()

# %%
graph_kernel = GraphEdgeKernel(
    feature_mat=D.X,
    base_kernel=kernel,
    # laplacian=L,
    lengthscale=2.3,
    variance=3.2,
    smoothness=6.1,
)

# %%
edge_indices = jnp.array(G.edges)

# %%
kernel(D.X, D.y)

# %%
meanf = gpx.mean_functions.Constant()
prior = gpx.gps.Prior(mean_function=meanf, kernel=graph_kernel)
likelihood = gpx.likelihoods.Bernoulli(num_datapoints=D.n)

# %%
posterior = prior * likelihood

# %%
D = gpx.Dataset(X=edge_indices, y=y.astype(jnp.float64))

# %%
optimiser = ox.adam(learning_rate=0.01)
opt_posterior, history = gpx.fit(
    model=posterior,
    # we use the negative lpd as we are minimising
    objective=lambda p, d: -gpx.objectives.log_posterior_density(p, d),
    train_data=D,
    optim=ox.adamw(learning_rate=0.01),
    num_iters=1000,
    key=key,
    trainable=Parameter,  # train all parameters (default behavior)
)