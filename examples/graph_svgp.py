%load_ext autoreload
%autoreload 2
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.lax import reduce_sum
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve

import optax as ox
import gpjax as gpx
from gpjax.parameters import Parameter

# --- your data loading (unchanged) ---
from src.data.graph import build_string_undirected_graph, subset_connected_with_labels
from src.data.labels import load_binary_labels
from src.fit import fit
from src.variation import GraphVariationalGaussian
from src.objective import elbo
from jax import config

import random
vertex_per_side = 20
n_edges_to_remove = 30
p = 0.8
key = jr.key(42)
G = nx.barbell_graph(vertex_per_side, 0)

random.seed(123)
[G.remove_edge(*i) for i in random.sample(list(G.edges), n_edges_to_remove)]

pos = nx.spring_layout(G, seed=123)  # positions for all nodes

nx.draw(
    G, pos, node_size=100, edge_color="black", with_labels=False
)
L = nx.laplacian_matrix(G).toarray()
x = jnp.arange(G.number_of_nodes()).reshape(-1, 1)

kernel = gpx.kernels.GraphKernel(
    laplacian=L,
    lengthscale=2.3,
    variance=3.2,
    smoothness=6.1,
)
prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=true_kernel)

fx = prior(x)
y = fx.sample(key=key, sample_shape=(1,)).reshape(-1, 1)

D = gpx.Dataset(X=x, y=y)
meanf = gpx.mean_functions.Constant()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
likelihood = gpx.likelihoods.Bernoulli(num_datapoints=D.n)
p = prior * likelihood
z = jnp.array(np.random.randint(low=1, high=D.n, size=(128, 1)), dtype=jnp.int64)
q = GraphVariationalGaussian(posterior=p, inducing_inputs=z)
elbo(q, D)