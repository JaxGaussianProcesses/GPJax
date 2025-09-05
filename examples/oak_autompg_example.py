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
#     display_name: docs
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Orthogonal Additive Kernel (OAK) - Auto MPG Dataset Example
#
# This notebook recreates the Auto MPG regression example from the original OAK paper using GPJax implementation.
# The Auto MPG dataset contains fuel efficiency data for various car models, making it ideal for demonstrating
# interpretable regression with feature importance analysis.
#
# **Dataset Features:**
# - **Target**: Miles per gallon (MPG)
# - **Features**: cylinders, displacement, horsepower, weight, acceleration, year, origin
#
# This example demonstrates:
# 1. **Real-world regression** on the classic Auto MPG dataset
# 2. **Feature importance analysis** using Sobol indices
# 3. **Model interpretability** through additive decomposition
# 4. **Performance comparison** with standard RBF kernel

# %%
# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax as ox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from examples.utils import use_mpl_style

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

key = jr.PRNGKey(42)

# Set the default style for plotting
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Data Loading and Preprocessing
#
# We'll load the Auto MPG dataset from OpenML, which is equivalent to the dataset used
# in the original paper. The dataset contains information about cars from the 1970s-80s
# with the goal of predicting fuel efficiency (MPG) based on various car characteristics.

# %%
print("Loading Auto MPG dataset from UCI repository...")

# Import the UCI ML repo package

# Fetch the Auto MPG dataset (ID: 9)
auto_mpg = fetch_ucirepo(id=9)

# Get data as pandas DataFrames
X_raw = auto_mpg.data.features
y_raw = auto_mpg.data.targets

# Print metadata
print("\nDataset Metadata:")
print(f"  Name: {auto_mpg.metadata['name']}")
print(f"  Abstract: {auto_mpg.metadata['abstract'][:200]}...")
print(f"  Dataset Size: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")

# Print variable information
print("\nVariable Information:")
for _, var_info in auto_mpg.variables.iterrows():
    print(f"  {var_info['name']}: {var_info['type']} - {var_info['description']}")

# Handle missing values (horsepower has some missing values in the original dataset)
# Remove rows with missing values
mask = ~(X_raw.isna().any(axis=1) | y_raw.isna().any(axis=1))
X_raw = X_raw[mask]
y_raw = y_raw[mask]

# Get feature names
feature_names = X_raw.columns.tolist()
print(f"\nFeatures used for modeling: {feature_names}")

print(f"Dataset shape: {X_raw.shape}")
print(f"Target range: {y_raw.values.min():.1f} - {y_raw.values.max():.1f} MPG")

# %%
# Convert to numpy arrays - use float64 for better numerical stability
X = X_raw.values.astype(np.float64)
y = y_raw.values.astype(np.float64).reshape(-1, 1)

# Split into train/test (80/20 split, same as original paper)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# %%
# Standardize features (important for GP kernels)
# Following the original paper's preprocessing approach
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to JAX arrays
X_train_jax = jnp.array(X_train_scaled)
X_test_jax = jnp.array(X_test_scaled)
y_train_jax = jnp.array(y_train_scaled)
y_test_jax = jnp.array(y_test_scaled)

print("Data preprocessing completed")
print(
    f"Feature ranges after scaling: {X_train_jax.min():.2f} to {X_train_jax.max():.2f}"
)

# %% [markdown]
# ## Data Visualization
#
# Let's visualize the relationships between each feature and the target (MPG) to understand
# the dataset structure and identify potential non-linear relationships.

# %%
# Visualize the data
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for i, feature_name in enumerate(feature_names):
    if i < len(feature_names):
        axes[i].scatter(
            X_train[:, i], y_train.squeeze(), alpha=0.6, s=30, color=cols[i % len(cols)]
        )
        axes[i].set_xlabel(feature_name.replace("_", " ").title())
        axes[i].set_ylabel("MPG")
        axes[i].set_title(f"MPG vs {feature_name.replace('_', ' ').title()}")
        axes[i].grid(True, alpha=0.3)

# Remove empty subplots
for i in range(len(feature_names), len(axes)):
    axes[i].remove()

plt.tight_layout()

# %% [markdown]
# ## Setting up the OAK Gaussian Process
#
# Now we'll create a GP with the Orthogonal Additive Kernel. Based on the original paper,
# we use appropriate hyperparameter settings for the Auto MPG dataset:
#
# - `lengthscales`: Individual lengthscales for each feature
# - `max_order`: Maximum interaction order (up to 2nd order interactions)
# - `input_means`: Set to zero for standardized data
# - `input_scales`: Set to one for standardized data
# - `variances`: Prior variances for each interaction order

# %%
n_features = X_train_jax.shape[1]

# Create the OAK kernel with more conservative hyperparameters for stability
oak_kernel = gpx.kernels.OrthogonalAdditiveKernel(
    lengthscales=[1.0] * n_features,  # Initial lengthscales
    max_order=2,  # Up to 2nd order interactions for stability
    input_means=[0.0] * n_features,  # Zero means for standardized data
    input_scales=[1.0] * n_features,  # Unit scales for standardized data
    variances=[1.0, 0.5, 0.25],  # Decreasing prior variances by order
)

# Create mean function and prior
mean_function = gpx.mean_functions.Zero()
oak_prior = gpx.gps.Prior(kernel=oak_kernel, mean_function=mean_function)

# Create likelihood and dataset
likelihood = gpx.likelihoods.Gaussian(num_datapoints=len(y_train_jax))
dataset = gpx.Dataset(X=X_train_jax, y=y_train_jax)

# Form the posterior
oak_posterior = oak_prior * likelihood

print(f"OAK Kernel created with {n_features} features")
print(f"Maximum interaction order: {oak_kernel.max_order}")
print(f"Variance parameters shape: {oak_kernel.variances.value.shape}")

# %% [markdown]
# ## Hyperparameter Optimization
#
# We optimize the kernel hyperparameters by maximizing the log marginal likelihood.
# This follows the same optimization approach as the original paper.


# %%
# Define loss function
def loss_fn(posterior, data):
    return -gpx.objectives.conjugate_mll(posterior, data)


print("Initial negative log marginal likelihood:", loss_fn(oak_posterior, dataset))

# Optimize hyperparameters using Adam optimizer with more conservative settings
print("Optimizing OAK hyperparameters...")
opt_posterior, history = gpx.fit(
    model=oak_posterior,
    objective=loss_fn,
    train_data=dataset,
    optim=ox.adam(learning_rate=1e-3),  # Smaller learning rate for stability
    num_iters=500,  # Fewer iterations
    key=key,
)

print("Final negative log marginal likelihood:", loss_fn(opt_posterior, dataset))
print(f"Optimized lengthscales: {opt_posterior.prior.kernel.lengthscales.value}")
print(f"Optimized variances: {opt_posterior.prior.kernel.variances.value}")

# %% [markdown]
# ## Making Predictions and Computing Performance Metrics
#
# Now we'll make predictions on the test set and compute performance metrics
# to compare with the original paper results.

# %%
# Make predictions on test data
latent_dist = opt_posterior.predict(X_test_jax, dataset)
predictive_dist = opt_posterior.likelihood(latent_dist)

# Get predictions in original scale
pred_mean_scaled = predictive_dist.mean.squeeze()
pred_std_scaled = jnp.sqrt(predictive_dist.variance.squeeze())

# Transform back to original scale
pred_mean = scaler_y.inverse_transform(pred_mean_scaled.reshape(-1, 1)).squeeze()
y_test_original = scaler_y.inverse_transform(y_test_scaled).squeeze()

# Compute performance metrics
rmse = jnp.sqrt(jnp.mean((pred_mean - y_test_original) ** 2))
mae = jnp.mean(jnp.abs(pred_mean - y_test_original))

# R-squared calculation
ss_res = jnp.sum((y_test_original - pred_mean) ** 2)
ss_tot = jnp.sum((y_test_original - jnp.mean(y_test_original)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\n=== OAK Performance on Auto MPG ===")
print(f"Test RMSE: {rmse:.3f} MPG")
print(f"Test MAE:  {mae:.3f} MPG")
print(f"Test R²:   {r2:.3f}")

# Compute negative log-likelihood (as in original paper)
nll = -jnp.mean(predictive_dist.log_prob(y_test_scaled.squeeze()))
print(f"Test NLL:  {nll:.4f}")

# %% [markdown]
# ## Computing Sobol Indices for Feature Importance
#
# The key advantage of OAK is automatic feature importance analysis through Sobol indices.
# We'll compute both grouped and detailed Sobol indices to understand which features
# and interactions are most important for predicting MPG.

# %%
# Compute Sobol indices using the new full decomposition
print("Computing Sobol indices for feature importance...")

sobol_indices = gpx.kernels.compute_sobol_indices(
    opt_posterior, dataset.X, dataset.y, use_full_decomposition=True, n_samples=1000
)

detailed_sobol = gpx.kernels.compute_detailed_sobol_indices(
    opt_posterior, dataset.X, dataset.y, n_samples=1000, use_full_decomposition=True
)

print("\n=== Sobol Indices (Feature Importance) ===")
for component_name, value in sobol_indices.items():
    print(f"  {component_name.replace('_', ' ').title()}: {value:.3f}")

print("\n=== Individual Feature Contributions ===")
for feature_name, value in detailed_sobol["main_effects"].items():
    feature_idx = int(feature_name.split("_")[-1])
    actual_name = feature_names[feature_idx]
    print(f"  {actual_name}: {value:.3f}")

print(f"\nTotal variance explained: {detailed_sobol['total_variance']:.3f}")

# %% [markdown]
# ## Detailed Additive Decomposition Analysis
#
# Using the new OrthogonalAdditiveGP class, we can decompose the posterior mean
# into individual additive components and analyze their contributions.

# %%
# Demonstrate the full decomposition with OrthogonalAdditiveGP
print("\n=== Full Additive Decomposition Analysis ===")

# Create OrthogonalAdditiveGP wrapper for detailed analysis
oak_gp = gpx.kernels.OrthogonalAdditiveGP(opt_posterior, dataset)

# Compute individual component Sobol indices
individual_sobol = oak_gp.compute_true_sobol_indices(n_samples=1000)

print("Individual Component Sobol Indices:")
for component_name, value in individual_sobol.items():
    if "main_effect" in component_name:
        feature_idx = int(component_name.split("_")[-1])
        actual_name = feature_names[feature_idx]
        print(f"  {actual_name}: {value:.4f}")
    elif "interaction" in component_name:
        print(f"  {component_name}: {value:.4f}")
    else:
        print(f"  {component_name}: {value:.4f}")

# %% [markdown]
# ## Comparison with Standard RBF Kernel
#
# To validate the performance of OAK, let's compare it with a standard RBF kernel GP,
# as done in the original paper.

# %%
print("\n=== Comparison with RBF Kernel ===")

# Create and optimize RBF GP for comparison
rbf_kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
rbf_prior = gpx.gps.Prior(kernel=rbf_kernel, mean_function=mean_function)
rbf_posterior = rbf_prior * likelihood

print("Optimizing RBF kernel...")
rbf_opt_posterior, rbf_history = gpx.fit(
    model=rbf_posterior,
    objective=loss_fn,
    train_data=dataset,
    optim=ox.adam(learning_rate=1e-2),
    num_iters=1000,
    key=key,
)

# RBF predictions
rbf_latent_dist = rbf_opt_posterior.predict(X_test_jax, dataset)
rbf_predictive_dist = rbf_opt_posterior.likelihood(rbf_latent_dist)
rbf_pred_mean_scaled = rbf_predictive_dist.mean.squeeze()

# Transform back to original scale
rbf_pred_mean = scaler_y.inverse_transform(
    rbf_pred_mean_scaled.reshape(-1, 1)
).squeeze()
rbf_rmse = jnp.sqrt(jnp.mean((rbf_pred_mean - y_test_original) ** 2))
rbf_nll = -jnp.mean(rbf_predictive_dist.log_prob(y_test_scaled.squeeze()))

print(f"RBF Test RMSE: {rbf_rmse:.3f} MPG")
print(f"RBF Test NLL:  {rbf_nll:.4f}")
print(f"OAK Test RMSE: {rmse:.3f} MPG")
print(f"OAK Test NLL:  {nll:.4f}")
print(f"RMSE Improvement: {((rbf_rmse - rmse) / rbf_rmse * 100):+.1f}%")
print(f"NLL Improvement:  {((rbf_nll - nll) / rbf_nll * 100):+.1f}%")

# %% [markdown]
# ## Visualization of Results
#
# Let's create comprehensive visualizations to understand the model performance
# and feature importance, similar to the analysis in the original paper.

# %%
# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Predictions vs True values
axes[0, 0].scatter(y_test_original, pred_mean, alpha=0.7, color=cols[0], label="OAK")
axes[0, 0].scatter(
    y_test_original, rbf_pred_mean, alpha=0.7, color=cols[1], label="RBF"
)
axes[0, 0].plot(
    [y_test_original.min(), y_test_original.max()],
    [y_test_original.min(), y_test_original.max()],
    "k--",
    alpha=0.8,
)
axes[0, 0].set_xlabel("True MPG")
axes[0, 0].set_ylabel("Predicted MPG")
axes[0, 0].set_title(
    f"Predictions Comparison\n(OAK RMSE: {rmse:.2f}, RBF RMSE: {rbf_rmse:.2f})"
)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals analysis
oak_residuals = pred_mean - y_test_original
rbf_residuals = rbf_pred_mean - y_test_original

axes[0, 1].scatter(
    pred_mean,
    oak_residuals,
    alpha=0.7,
    color=cols[0],
    label=f"OAK (σ={jnp.std(oak_residuals):.2f})",
)
axes[0, 1].scatter(
    rbf_pred_mean,
    rbf_residuals,
    alpha=0.7,
    color=cols[1],
    label=f"RBF (σ={jnp.std(rbf_residuals):.2f})",
)
axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.8)
axes[0, 1].set_xlabel("Predicted MPG")
axes[0, 1].set_ylabel("Residuals")
axes[0, 1].set_title("Residuals Analysis")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature importance (Sobol indices by order)
sobol_names = list(sobol_indices.keys())
sobol_values = list(sobol_indices.values())
display_names = [name.replace("_", " ").title() for name in sobol_names]

axes[0, 2].bar(display_names, sobol_values, color=cols[: len(sobol_values)], alpha=0.7)
axes[0, 2].set_ylabel("Sobol Index")
axes[0, 2].set_title("Feature Importance by Interaction Order")
axes[0, 2].tick_params(axis="x", rotation=45)
axes[0, 2].grid(True, alpha=0.3)

# 4. Individual feature importance
feature_importances = []
feature_labels = []
for i, feature_name in enumerate(feature_names):
    if f"feature_{i}" in detailed_sobol["main_effects"]:
        feature_importances.append(detailed_sobol["main_effects"][f"feature_{i}"])
        feature_labels.append(feature_name.replace("_", " ").title())

axes[1, 0].bar(
    feature_labels,
    feature_importances,
    color=cols[: len(feature_importances)],
    alpha=0.7,
)
axes[1, 0].set_ylabel("Sobol Index")
axes[1, 0].set_title("Individual Feature Importance")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# 5. Kernel variance parameters by order
variance_orders = [
    f"Order {i}" for i in range(len(opt_posterior.prior.kernel.variances.value))
]
variance_values = opt_posterior.prior.kernel.variances.value

axes[1, 1].bar(
    variance_orders, variance_values, color=cols[: len(variance_values)], alpha=0.7
)
axes[1, 1].set_ylabel("Variance Parameter")
axes[1, 1].set_title("OAK Kernel Variance by Interaction Order")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].grid(True, alpha=0.3)

# 6. Training curves comparison
axes[1, 2].plot(-jnp.array(history), label="OAK", color=cols[0])
axes[1, 2].plot(-jnp.array(rbf_history), label="RBF", color=cols[1])
axes[1, 2].set_xlabel("Iteration")
axes[1, 2].set_ylabel("Log Marginal Likelihood")
axes[1, 2].set_title("Training Convergence")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()

# %% [markdown]
# ## Analysis of Most Important Interactions
#
# Let's identify and analyze the most important pairwise interactions discovered by OAK.

# %%
print("\n=== Analysis of Important Interactions ===")

# Get the most important interactions
interaction_importance = {}
for comp_name, value in individual_sobol.items():
    if "interaction" in comp_name and value > 0.001:  # Only significant interactions
        # Parse interaction indices
        indices_str = comp_name.replace("interaction_", "")
        indices = list(map(int, indices_str.split("_")))

        if len(indices) == 2:  # Pairwise interactions
            feature_pair = f"{feature_names[indices[0]]} × {feature_names[indices[1]]}"
            interaction_importance[feature_pair] = value

# Sort by importance
sorted_interactions = sorted(
    interaction_importance.items(), key=lambda x: x[1], reverse=True
)

print("Most Important Pairwise Interactions:")
for interaction, importance in sorted_interactions[:5]:
    print(f"  {interaction}: {importance:.4f}")

# %% [markdown]
# ## Interpretation and Summary
#
# Let's interpret the results and compare with automotive domain knowledge.

# %%
print("\n=== Model Interpretation and Summary ===")

# Analyze variance contributions by interaction order
total_var = jnp.sum(opt_posterior.prior.kernel.variances.value)
print("Variance Distribution by Interaction Order:")
for i, var in enumerate(opt_posterior.prior.kernel.variances.value):
    percentage = (var / total_var) * 100
    if i == 0:
        print(f"  Constant term: {percentage:.1f}%")
    elif i == 1:
        print(f"  Main effects: {percentage:.1f}%")
    elif i == 2:
        print(f"  Pairwise interactions: {percentage:.1f}%")
    else:
        print(f"  Order-{i} interactions: {percentage:.1f}%")

# Most important features
sorted_features = sorted(
    zip(feature_names, feature_importances, strict=False),
    key=lambda x: x[1],
    reverse=True,
)
print("\nMost Important Features for MPG Prediction:")
for feature, importance in sorted_features:
    print(f"  {feature.replace('_', ' ').title()}: {importance:.4f}")

print("\n=== Final Results Summary ===")
print(f"Dataset: Auto MPG ({X_train.shape[0]} train, {X_test.shape[0]} test samples)")
print("OAK Model Performance:")
print(f"  - RMSE: {rmse:.3f} MPG")
print(f"  - R²: {r2:.3f}")
print(f"  - NLL: {nll:.4f}")
print("RBF Model Performance:")
print(f"  - RMSE: {rbf_rmse:.3f} MPG")
print(f"  - NLL: {rbf_nll:.4f}")
print(
    f"OAK Improvement: {((rbf_rmse - rmse) / rbf_rmse * 100):+.1f}% RMSE, {((rbf_nll - nll) / rbf_nll * 100):+.1f}% NLL"
)

print("\nInterpretability Benefits:")
print(
    f"  - {len([f for f in feature_importances if f > 0.05])} highly important features identified"
)
print(
    f"  - {len([i for i in interaction_importance.values() if i > 0.01])} significant pairwise interactions found"
)
print(f"  - Main effects explain {sobol_indices['main_effects']:.1%} of variance")
print(
    f"  - Interactions explain {sobol_indices.get('order_2_interactions', 0):.1%} of variance"
)

# %% [markdown]
# ## Conclusion
#
# This example successfully demonstrates the Orthogonal Additive Kernel (OAK) on the Auto MPG dataset,
# recreating the analysis from the original paper using the GPJax implementation.
#
# ### Key Findings:
#
# 1. **Performance**: OAK achieves competitive or superior predictive performance compared to RBF
# 2. **Interpretability**: Automatic feature importance ranking reveals which car characteristics most affect MPG
# 3. **Additive Structure**: The model decomposes MPG prediction into interpretable additive components
# 4. **Interaction Discovery**: OAK identifies important feature interactions automatically
#
# ### Practical Insights:
#
# The feature importance analysis aligns with automotive domain knowledge:
# - **Weight** and **displacement** typically have the strongest negative correlation with fuel efficiency
# - **Year** often shows positive correlation (improving technology over time)
# - **Horsepower** vs **weight** interactions are automotive-relevant
#
# ### Advantages of OAK:
#
# 1. **Automatic Feature Selection**: Identifies which features matter most
# 2. **Interaction Discovery**: Finds important feature combinations without manual specification
# 3. **Interpretable Decomposition**: Each component has a clear meaning
# 4. **Uncertainty Quantification**: Provides prediction uncertainty like any GP
#
# The GPJax implementation provides a modern, efficient way to apply OAK to real-world regression problems
# while maintaining the interpretability benefits that make it valuable for scientific and industrial applications.

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'GPJax Team'
