# %% [markdown]
# # Graph Edge Kernels — medium random graph (~2000 edges)

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
from sklearn.model_selection import train_test_split

import gpjax as gpx
from gpjax.kernels.non_euclidean.graph_edge import GraphEdgeKernel
from gpjax.parameters import Parameter

# %% [markdown]
# ## Configuration
# %%
SEED = 123
np.random.seed(SEED)
random.seed(SEED)
key = jr.key(42)

# %% [markdown]
# ## Construct medium-sized random graph (~2000 edges)
# %%
n_nodes = 150
target_edges = 3000
p_edge = target_edges / (n_nodes * (n_nodes - 1) / 2)
G = nx.erdos_renyi_graph(n_nodes, p_edge, seed=SEED)

while G.number_of_edges() > target_edges:
    u, v = random.choice(list(G.edges()))
    G.remove_edge(u, v)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
# %%
pos = nx.spring_layout(G, seed=SEED)
plt.figure(figsize=(6, 5))
nx.draw(G, pos, node_size=40, edge_color="black", with_labels=False)
plt.title(
    f"Random graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
)
plt.show()

# %% [markdown]
# ## Node features
# %%
node_feature_dim = 20
node_feature_matrix = np.random.uniform(
    low=0.5, high=13.3, size=(n_nodes, node_feature_dim)
).astype(np.float64)

# %% [markdown]
# ## Prepare edges and labels
# %%
edge_list = jnp.array(G.edges).astype(jnp.int64)
num_edges = edge_list.shape[0]
pos_frac = 0.5
n_pos = int(pos_frac * num_edges)
labels = np.array([1] * n_pos + [0] * (num_edges - n_pos), dtype=np.float64)
np.random.shuffle(labels)
labels_jnp = jnp.array(labels).reshape(-1, 1).astype(jnp.float64)
# %% [markdown]
# ## Train / Test split
# %%
edge_idx = np.arange(num_edges)
train_idx, test_idx = train_test_split(
    edge_idx, test_size=0.2, random_state=SEED, stratify=labels
)

edge_train = edge_list[train_idx]
edge_test = edge_list[test_idx]
y_train = labels_jnp[train_idx]
y_test = labels_jnp[test_idx]

print(f"Training edges: {len(train_idx)}, Test edges: {len(test_idx)}")

# %% [markdown]
# ## Model definition
# %%
base_kernel = gpx.kernels.RBF()
graph_kernel = GraphEdgeKernel(feature_mat=node_feature_matrix, base_kernel=base_kernel)
meanf = gpx.mean_functions.Constant()
prior = gpx.gps.Prior(mean_function=meanf, kernel=graph_kernel)
likelihood = gpx.likelihoods.Bernoulli(num_datapoints=len(train_idx))
posterior = prior * likelihood

# %% [markdown]
# ## Train model
# %%
D_train = gpx.Dataset(X=jnp.array(edge_train), y=y_train)
D_test = gpx.Dataset(X=jnp.array(edge_test), y=y_test)

optimiser = ox.adamw(learning_rate=0.1)
num_iters = 2000

opt_posterior, history = gpx.fit(
    model=posterior,
    objective=lambda p, d: -gpx.objectives.log_posterior_density(p, d),
    train_data=D_train,
    optim=optimiser,
    num_iters=num_iters,
    key=key,
    trainable=Parameter,
)

# %% [markdown]
# ## Predictions on test edges
# %%
pred_dist = opt_posterior.likelihood(opt_posterior(edge_test, D_train))
pred_mean = pred_dist.mean

y_prob_np = np.array(pred_mean)
y_test_np = np.array(y_test)

pred_labels = jnp.where(y_prob_np > 0.5, 1, 0)
auc = roc_auc_score(y_test_np, y_prob_np)
acc = accuracy_score(y_test_np, pred_labels)
f1 = f1_score(y_test_np, pred_labels)
print(f"Test ROC-AUC: {auc:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")

# %% [markdown]
# ## Plot ROC curve
# %%
fpr, tpr, _ = roc_curve(y_test_np, y_prob_np)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Test Edges")
plt.grid(True)
plt.show()
