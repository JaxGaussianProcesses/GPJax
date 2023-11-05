# # ---
# # jupyter:
# #   jupytext:
# #     cell_metadata_filter: -all
# #     custom_cell_magics: kql
# #     text_representation:
# #       extension: .py
# #       format_name: percent
# #       format_version: '1.3'
# #       jupytext_version: 1.11.2
# #   kernelspec:
# #     display_name: gpjax
# #     language: python
# #     name: python3
# # ---

# # %% [markdown]
# # # Gaussian Process Latent Variable Models
# #
# # This notebook shows how to implement a Gaussian process latent variable model (GPLVM) [(Lawrence (2003))](https://proceedings.neurips.cc/paper/2003/hash/9657c1fffd38824e5ab0472e022e577e-Abstract.html). The GPLVM is then demonstrated on Kaggle's [countries of the world](https://www.kaggle.com/datasets/fernandol/countries-of-the-world) dataset to infer a low-dimensional representation of the countries.

# # %%
# from jax import config

# config.update("jax_enable_x64", True)

# import jax
# import jax.numpy as jnp
# import jax.random as jr
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from jaxtyping import install_import_hook
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# from dataclasses import dataclass
# import optax as ox
# import seaborn as sns

# with install_import_hook("gpjax", "beartype.beartype"):
#     import gpjax as gpx
#     from gpjax.base import param_field
#     from gpjax.typing import ScalarFloat

# key = jr.PRNGKey(123)
# plt.style.use("https://raw.githubusercontent.com/JaxGaussianProcesses/static/main/configs/gpjax.mplstyle")
# cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# # %% [markdown]
# # ## Data
# #
# # We'll begin by importing the countries of the world dataset. For each country we have the following 17 features
# # * Population
# # * Area (squared miles)
# # * Population Density (per square mile)
# # * Coastline (coast/area ratio)
# # * Net migration
# # * Infant mortality (per 1000 births)
# # * Literacy (%)
# # * Phones (per 1000)
# # * Arable (%)
# # * Crops (%)
# # * Other (%)
# # * Climate
# # * Birthrate
# # * Deathrate
# # * Agriculture
# # * Industry
# # * Service
# #
# # We drop GDP from the data as we will use it to colour the latent space to visually assess whether the GPLVM is able to infer a latent representation of the countries' GDP.
# #
# # Finally, we'll standardise each feature to have zero mean and unit variance.

# # %%
# world_data = pd.read_csv(
#     "https://raw.githubusercontent.com/JaxGaussianProcesses/static/main/data/countries_of_the_world.csv", decimal=","
# ).dropna()

# label = world_data[["GDP ($ per capita)"]]
# features = world_data.drop(["Country", "Region", "GDP ($ per capita)"], axis="columns")

# features = StandardScaler().fit_transform(features)

# # %% [markdown]
# #
# # ## Model specification
# #
# # For an observed dataset $\mathbf{Y}\in\mathbb{R}^{N\times D}$, the task of a GPLVM is to infer a latent representation $\mathbf{X}\in\mathbb{R}^{N\times Q}$ where $Q\ll D$. This is achieved through a set of $Q$ Gaussian process $(f_1, f_2, \ldots, f_Q)$ where each $f_q(\cdot)\sim\mathcal{GP}(m(\cdot), k(\cdot, \cdot))$ is a mapping from the $q^{\text{th}}$ latent dimension latent space to the observed space. The hierarchical model can then be written as
# # $$\begin{align}
# # p(\mathbf{X}) & = \prod_{n=1}^N \mathcal{N}(\mathbf{x}_{n}\mid\mathbf{0}, \mathbf{I}_Q) \\
# # p(\mathbf{f}\mid \mathbf{X}, \mathbf{\theta}) & = \prod_{d=1}^{D} \mathcal{N}(\mathbf{f}_{d}\mid \mathbf{0}, \mathbf{K}_{\mathbf{ff}}) \\
# # p(\mathbf{Y}\mid\mathbf{f}, \mathbf{X}) & = \prod_{n=1}^{N}\prod_{d=1}^{D}\mathcal{N}(y_{n, d}\mid f_d(\mathbf{x}_n), \sigma^2)
# # \end{align}
# # $$
# # where $\mathbf{f}_d = f_d(\mathbf{X})$. In the GPLVM implemented with GPJax, we learn
# # a MAP estimate of the latent coordinates that enables analytical marginalisation of
# # the latent GP.

# # %%
# latent_dim = 2
# principal_components = PCA(n_components=latent_dim).fit_transform(features)
# initial_X = jnp.asarray(principal_components)

# # %%
# meanf = gpx.mean_functions.Zero()
# prior = gpx.Prior(
#     mean_function=meanf,
#     kernel=gpx.kernels.Matern52(active_dims=list(range(latent_dim))),
# )
# likelihood = gpx.likelihoods.Gaussian(num_datapoints=initial_X.shape[0])

# latent_processes = prior * likelihood


# # %% [markdown]
# # We'll now define the GPLVM using the `gpx.Module` provided in GPJax. Unlike in [GP regression](./regression.py), we have to augment the model to contain not only the posterior distribution, but also the latent variables. Fortunately, this is easily achieved by adding an additional line to the model's field list and specifying it as a `param_field`. This will ensure that the latent variables are optimised alongside the GP's hyperparameters.


# # %%
# @dataclass
# class GPLVM(gpx.Module):
#     latent_process: gpx.gps.ConjugatePosterior
#     latent_variables: jax.Array = param_field()


# model = GPLVM(latent_process=latent_processes, latent_variables=initial_X)


# # %% [markdown]
# # ## Optimisation
# #
# # We can now optimise the parameters of our model. We achieve this by optimising the marginal log-likelihood as used in [regression notebook](./regression.py) where we now compute its value for each of $D$ observed dimensions and sum them together. We'll also use the `jax.jit` decorator to JIT compile the function to accelerate optimisation.


# # %%
# @dataclass
# class GPLVM_MAP(gpx.objectives.AbstractObjective):
#     def step(
#         self,
#         posterior: GPLVM,  # noqa: F821
#         observed_data: jax.Array,  # noqa: F821
#     ) -> ScalarFloat:
#         Z = posterior.latent_variables

#         def _single_mll(y: jax.Array):
#             y = y[:, None]
#             D = gpx.Dataset(X=Z, y=y, mask=None)
#             objective = gpx.objectives.ConjugateMLL(negative=True)
#             return objective.step(posterior.latent_process, D)

#         summed_mll = jnp.sum(jax.vmap(_single_mll)(observed_data))
#         return self.constant * summed_mll


# objective = jax.jit(GPLVM_MAP().step)

# # %%
# Y = jnp.asarray(features).T

# num_iters = 2000

# schedule = ox.warmup_cosine_decay_schedule(
#     init_value=0.0,
#     peak_value=0.1,
#     warmup_steps=75,
#     decay_steps=num_iters // 2,
#     end_value=0.001,
# )

# optim = ox.adamw(learning_rate=0.01)
# state = optim.init(model)


# def loss(model) -> ScalarFloat:
#     model = model.stop_gradient()
#     return objective(model.constrain(), Y)


# iter_keys = jr.split(key, num_iters)


# def step(carry, key):
#     model, opt_state = carry

#     loss_val, loss_gradient = jax.value_and_grad(loss)(model)
#     updates, opt_state = optim.update(loss_gradient, opt_state, model)
#     model = ox.apply_updates(model, updates)

#     carry = model, opt_state
#     return carry, loss_val


# # Optimisation loop.
# (model, _), history = gpx.scan.vscan(step, (model, state), (iter_keys), unroll=1)

# # Constrained space.
# model = model.constrain()

# # +
# Z = model.latent_variables
# results = pd.DataFrame(jnp.hstack((Z, label.values)), columns=["x", "y", "label"])

# # %% [markdown]
# # ## Latent space visualisation
# #
# # With optimisation complete, we can now visualise our latent space. To do this, we'll
# # simply plot the 2D coordinate that has been learned for each observation and colour
# # it by the country's GDP.

# # %%
# fig, ax = plt.subplots()

# scatter = ax.scatter(
#     results["x"],
#     results["y"],
#     c=results["label"],
#     cmap="viridis",
#     norm=mpl.colors.LogNorm(),
# )
# cbar = fig.colorbar(scatter, ax=ax)
# cbar.set_label("log-GDP")
# ax.set(
#     xlabel="Latent dimension 1",
#     ylabel="Latent dimension 2",
#     title="Latent space of GDP data",
# )
# # %% [markdown]
# # ## System configuration

# # %%
# # %reload_ext watermark
# # %watermark -n -u -v -iv -w -a 'Thomas Pinder'
