# %%
import jax.numpy as jnp
import datasets as ds
import gpjax as gpx
from jax import jit
import optax as ox
import jax.random as jr
from jaxtyping import PyTree
import matplotlib.pyplot as plt

# %% [markdown]
# Now load a graph dataset and pad it

# %%
gd = ds.load_dataset("graphs-datasets/AQSOL")

gd = gd.map(
    lambda x: {
        "num_edges": len(x["edge_index"][0]),
    }
)
gd.set_format("jax")

max_num_edges = max([gd[i]["num_edges"].max() for i in gd])
max_num_nodes = max([gd[i]["num_nodes"].max() for i in gd])

small_gd = (
    gd["train"]
    .select(range(100))
    .map(
        lambda x: {
            "num_edges": len(x["edge_index"][0]),
        }
    )
)


def pad_edge_attr_node_feat(x):
    nf = (
        jnp.zeros(max_num_nodes).at[: len(x["node_feat"])].set(x["node_feat"].squeeze())
    )
    ea = (
        jnp.zeros(max_num_edges).at[: len(x["edge_attr"])].set(x["edge_attr"].squeeze())
    )
    return {"node_feat": nf, "edge_attr": ea}


small_gd = small_gd.map(pad_edge_attr_node_feat)

# prepare the dataset for GPjax
D = gpx.Dataset(X={i: small_gd[i] for i in ("node_feat", "edge_attr")}, y=small_gd["y"])

# %% [markdown]
# Now define a naive Graph kernel that takes node and edge features


# %%
class GraphKern(gpx.AbstractKernel):
    def __call__(self, x1: PyTree, x2: PyTree, **kwargs):
        return gpx.kernels.RBF()(x1["node_feat"], x2["node_feat"]) + gpx.kernels.RBF()(
            x1["edge_attr"], x2["edge_attr"]
        )


# %% [markdown]
# And we're ready to fit a model!

# %%
meanf = gpx.mean_functions.Zero()
prior = gpx.Prior(mean_function=meanf, kernel=GraphKern())
likelihood = gpx.Gaussian(num_datapoints=D.n)
negative_mll = jit(gpx.objectives.ConjugateMLL(negative=True))
likelihood = gpx.Gaussian(num_datapoints=D.n)
posterior = prior * likelihood

opt_posterior, mll_history = gpx.fit(
    model=posterior,
    objective=negative_mll,
    train_data=D,
    optim=ox.adam(learning_rate=0.01),
    num_iters=600,
    safe=True,
    key=jr.PRNGKey(0),
)

# %%
plt.plot(mll_history)

# %%
