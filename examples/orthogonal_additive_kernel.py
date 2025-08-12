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
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: docs
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Orthogonal Additive Kernel (OAK) - Interpretable Gaussian Processes

# %% [markdown]
# In this notebook, we demonstrate the Orthogonal Additive Kernel (OAK) from Lu et al. (2022),
# which enables interpretable additive function decomposition in Gaussian processes.
# The OAK kernel provides:
#
# 1. **Identifiable additive decomposition**: $f(x) = f_0 + \sum_i f_i(x_i) + \sum_{i<j} f_{ij}(x_i, x_j) + ...$
# 2. **Orthogonality constraints**: Each component is orthogonal under the input measure
# 3. **Sobol indices**: Automatic feature importance analysis
# 4. **Efficient computation**: Newton-Girard algorithm for tractable inference

# %%
# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Float,
    install_import_hook,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
import pandas as pd

from examples.utils import use_mpl_style
from gpjax.parameters import Static
from gpjax.typing import Array

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

key = jr.PRNGKey(42)

# set the default style for plotting
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## What is the Orthogonal Additive Kernel?
#
# The OAK kernel decomposes functions into orthogonal additive components:
#
# $$f(\mathbf{x}) = f_0 + \sum_{u \subseteq \{1,\ldots,d\}} f_u(\mathbf{x}_u)$$
#
# where each component $f_u$ satisfies the orthogonality constraint:
# $$\int f_u(\mathbf{x}_u) p_i(x_i) dx_i = 0 \quad \forall i \in u$$
#
# This orthogonality ensures that each component represents a unique aspect of the
# function, making the decomposition **identifiable** and **interpretable**.
#
# The kernel uses the Newton-Girard algorithm to efficiently compute interaction terms
# in polynomial time rather than exponential time, making it tractable for practical use.

# %% [markdown]
# ## Generating Synthetic Additive Data
#
# Let's start with a synthetic dataset where we know the true additive structure.
# We'll use the Friedman #1 function, which has known additive and interaction terms:
#
# $$f(x_1, x_2, x_3, x_4, x_5) = 10 \sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5$$


# %%
# Generate synthetic additive data
def friedman_function(X):
    """
    Friedman #1 function with known additive structure.
    f(x1,x2,x3,x4,x5) = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5
    """
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    return (
        10 * jnp.sin(jnp.pi * x1 * x2)  # interaction term x1*x2
        + 20 * (x3 - 0.5) ** 2  # non-linear main effect x3
        + 10 * x4  # linear main effect x4
        + 5 * x5  # linear main effect x5
    )


# Generate training data
n_train = 150
key, subkey = jr.split(key)

# Generate data directly using uniform random samples (better for OAK)
X_train = jr.uniform(subkey, (n_train, 5), minval=0, maxval=1)
y_train = friedman_function(X_train).reshape(-1, 1)

# Add small amount of noise
noise_key, subkey = jr.split(subkey)
noise = 0.5 * jr.normal(noise_key, y_train.shape)
y_train = y_train + noise

X_normalized = X_train  # Already in [0, 1]

# Generate test data
n_test = 100
key, subkey = jr.split(key)
X_test_raw = jr.uniform(subkey, (n_test, 5), minval=0, maxval=1)
X_test = X_test_raw
y_test = friedman_function(X_test).reshape(-1, 1)

print(f"Training data shape: X={X_normalized.shape}, y={y_train.shape}")
print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

# %%
# Visualize the training data
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for i in range(5):
    axes[i].scatter(
        X_normalized[:, i], y_train.squeeze(), alpha=0.6, s=30, color=cols[i]
    )
    axes[i].set_xlabel(f"$x_{i + 1}$")
    axes[i].set_ylabel("$y$")
    axes[i].set_title(f"Response vs Feature {i + 1}")

# Remove the empty subplot
axes[5].remove()
plt.tight_layout()

# %% [markdown]
# ## Setting up the OAK Gaussian Process
#
# Now let's create a GP with the OAK kernel. The key parameters are:
#
# - `lengthscales`: RBF lengthscales for each input dimension
# - `max_order`: Maximum interaction order (1=main effects only, 2=include pairwise interactions, etc.)
# - `input_means`: Mean of Gaussian input density (default: zeros)
# - `input_scales`: Scale of Gaussian input density (default: ones)
# - `variances`: Prior variance for each interaction order

# %%
# Create the OAK kernel with more conservative parameters
oak_kernel = gpx.kernels.OrthogonalAdditiveKernel(
    lengthscales=[0.3, 0.3, 0.3, 0.3, 0.3],  # Smaller lengthscales for better fit
    max_order=2,  # Include up to 2nd order interactions
    input_means=[0.5, 0.5, 0.5, 0.5, 0.5],  # Gaussian input means
    input_scales=[0.3, 0.3, 0.3, 0.3, 0.3],  # Smaller input scales
)

# Create mean function and prior
mean_function = gpx.mean_functions.Zero()
oak_prior = gpx.gps.Prior(kernel=oak_kernel, mean_function=mean_function)

# Create likelihood and dataset
likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_train)
dataset = gpx.Dataset(X=X_normalized, y=y_train)

# Form the posterior
oak_posterior = oak_prior * likelihood

print("OAK Kernel parameters:")
print(f"- Number of dimensions: {oak_kernel.n_dims}")
print(f"- Maximum interaction order: {oak_kernel.max_order}")
print(f"- Variance parameters shape: {oak_kernel.variances.value.shape}")

# %% [markdown]
# ## Hyperparameter Optimization
#
# Let's optimize the kernel hyperparameters by maximizing the log marginal likelihood:


# %%
# Optimize hyperparameters
def loss_fn(posterior, data):
    return -gpx.objectives.conjugate_mll(posterior, data)


print("Initial negative log marginal likelihood:", loss_fn(oak_posterior, dataset))

# Optimize using Adam with more conservative learning rate
opt_posterior, history = gpx.fit(
    model=oak_posterior,
    objective=loss_fn,
    train_data=dataset,
    optim=ox.adam(learning_rate=5e-3),
    num_iters=500,
    key=key,
)

print("Optimized negative log marginal likelihood:", loss_fn(opt_posterior, dataset))
print(f"Final lengthscales: {opt_posterior.prior.kernel.lengthscales.value}")
print(f"Final variances: {opt_posterior.prior.kernel.variances.value}")

# %% [markdown]
# ## Making Predictions and Computing Sobol Indices
#
# Now we can make predictions and compute Sobol indices for feature importance analysis:

# %%
# Make predictions on test data
latent_dist = opt_posterior.predict(X_test, dataset)
predictive_dist = opt_posterior.likelihood(latent_dist)

pred_mean = predictive_dist.mean.squeeze()
pred_std = jnp.sqrt(predictive_dist.variance.squeeze())

# Compute root mean squared error
rmse = jnp.sqrt(jnp.mean((pred_mean - y_test.squeeze()) ** 2))
print(f"Test RMSE: {rmse:.3f}")

# Compute Sobol indices for feature importance
sobol_indices = gpx.kernels.compute_sobol_indices(opt_posterior, dataset.X, dataset.y)
detailed_sobol = gpx.kernels.compute_detailed_sobol_indices(
    opt_posterior, dataset.X, dataset.y, n_samples=500
)

print("\nSobol Indices (Feature Importance):")
for component_name, value in sobol_indices.items():
    print(f"  {component_name}: {value:.3f}")

print(f"\nTotal variance explained: {detailed_sobol['total_variance']:.3f}")
print("Individual feature contributions:")
for feature_name, value in detailed_sobol["main_effects"].items():
    print(f"  {feature_name}: {value:.3f}")

# %% [markdown]
# ## Visualization of Results

# %%
# Plot predictions vs true values
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Prediction vs true scatter plot
axes[0].scatter(y_test.squeeze(), pred_mean, alpha=0.7, color=cols[0])
axes[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    alpha=0.8,
    label="Perfect prediction",
)
axes[0].set_xlabel("True values")
axes[0].set_ylabel("Predicted values")
axes[0].set_title(f"OAK Predictions (RMSE={rmse:.3f})")
axes[0].legend()

# Sobol indices bar plot
indices_values = list(sobol_indices.values())
indices_names = list(sobol_indices.keys())
# Clean up names for display
display_names = [name.replace("_", " ").title() for name in indices_names]

axes[1].bar(display_names, indices_values, color=cols[: len(indices_values)], alpha=0.7)
axes[1].set_ylabel("Sobol Index")
axes[1].set_title("Feature Importance (Sobol Indices)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()

# %% [markdown]
# ## Comparison with Standard RBF Kernel
#
# Let's compare the OAK kernel performance with a standard RBF kernel:

# %%
# Create and optimize an RBF GP for comparison
rbf_kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
rbf_prior = gpx.gps.Prior(kernel=rbf_kernel, mean_function=mean_function)
rbf_posterior = rbf_prior * likelihood

# Optimize RBF hyperparameters
print("Optimizing RBF kernel...")
rbf_opt_posterior, rbf_history = gpx.fit(
    model=rbf_posterior,
    objective=loss_fn,
    train_data=dataset,
    optim=ox.adam(learning_rate=5e-3),
    num_iters=500,
    key=key,
)

# RBF predictions
rbf_latent_dist = rbf_opt_posterior.predict(X_test, dataset)
rbf_predictive_dist = rbf_opt_posterior.likelihood(rbf_latent_dist)
rbf_pred_mean = rbf_predictive_dist.mean.squeeze()
rbf_rmse = jnp.sqrt(jnp.mean((rbf_pred_mean - y_test.squeeze()) ** 2))

print(f"RBF Test RMSE: {rbf_rmse:.3f}")
print(f"OAK Test RMSE: {rmse:.3f}")
print(f"Improvement: {((rbf_rmse - rmse) / rbf_rmse * 100):+.1f}%")

# %% [markdown]
# ## Detailed Comparison and Interpretability Analysis

# %%
# Create a comprehensive comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Prediction comparison
axes[0, 0].scatter(y_test.squeeze(), pred_mean, alpha=0.7, color=cols[0], label="OAK")
axes[0, 0].scatter(
    y_test.squeeze(), rbf_pred_mean, alpha=0.7, color=cols[1], label="RBF"
)
axes[0, 0].plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", alpha=0.8
)
axes[0, 0].set_xlabel("True values")
axes[0, 0].set_ylabel("Predicted values")
axes[0, 0].set_title("Prediction Comparison")
axes[0, 0].legend()

# Plot 2: Residuals comparison
oak_residuals = pred_mean - y_test.squeeze()
rbf_residuals = rbf_pred_mean - y_test.squeeze()

axes[0, 1].scatter(
    pred_mean,
    oak_residuals,
    alpha=0.7,
    color=cols[0],
    label=f"OAK (σ={jnp.std(oak_residuals):.3f})",
)
axes[0, 1].scatter(
    rbf_pred_mean,
    rbf_residuals,
    alpha=0.7,
    color=cols[1],
    label=f"RBF (σ={jnp.std(rbf_residuals):.3f})",
)
axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.8)
axes[0, 1].set_xlabel("Predicted values")
axes[0, 1].set_ylabel("Residuals")
axes[0, 1].set_title("Residuals Analysis")
axes[0, 1].legend()

# Plot 3: Feature importance (OAK only)
axes[1, 0].bar(
    display_names, indices_values, color=cols[: len(indices_values)], alpha=0.7
)
axes[1, 0].set_ylabel("Sobol Index")
axes[1, 0].set_title("OAK Feature Importance")
axes[1, 0].tick_params(axis="x", rotation=45)

# Plot 4: Kernel variance parameters
variance_orders = [
    f"Order {i}" for i in range(len(opt_posterior.prior.kernel.variances.value))
]
variance_values = opt_posterior.prior.kernel.variances.value

axes[1, 1].bar(
    variance_orders, variance_values, color=cols[: len(variance_values)], alpha=0.7
)
axes[1, 1].set_ylabel("Variance Parameter")
axes[1, 1].set_title("OAK Interaction Order Importance")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()

# %% [markdown]
# ## Understanding the Additive Decomposition
#
# One key advantage of the OAK kernel is that it provides an interpretable decomposition
# of the function into additive components. While extracting the exact components requires
# additional computation (beyond the scope of this example), the Sobol indices give us
# insight into the relative importance of different interaction orders and features.

# %%
# Analyze the learned kernel parameters
print("=== OAK Kernel Analysis ===")
print(f"Optimized lengthscales: {opt_posterior.prior.kernel.lengthscales.value}")
print(f"Optimized variances: {opt_posterior.prior.kernel.variances.value}")
print(f"Input means: {opt_posterior.prior.kernel.input_means.value}")
print(f"Input scales: {opt_posterior.prior.kernel.input_scales.value}")

# Interpret the variance parameters
print("\n=== Interaction Order Analysis ===")
total_var = jnp.sum(opt_posterior.prior.kernel.variances.value)
for i, var in enumerate(opt_posterior.prior.kernel.variances.value):
    percentage = (var / total_var) * 100
    if i == 0:
        print(f"Constant term: {percentage:.1f}% of total variance")
    elif i == 1:
        print(f"Main effects: {percentage:.1f}% of total variance")
    else:
        print(f"Order-{i} interactions: {percentage:.1f}% of total variance")

print("\n=== Comparison Summary ===")
print(f"OAK RMSE: {rmse:.4f}")
print(f"RBF RMSE: {rbf_rmse:.4f}")
print(f"Performance improvement: {((rbf_rmse - rmse) / rbf_rmse * 100):+.1f}%")

# %% [markdown]
# ## Real-World Example: Synthetic Additive Function
#
# Let's demonstrate the OAK kernel on a more complex synthetic dataset with known
# additive structure to show how the kernel can identify important features and interactions:


# %%
# Create a more complex synthetic additive function
def complex_additive_function(X):
    """
    Complex additive function with multiple types of effects:
    - Linear effects: x1, x4
    - Nonlinear effects: sin(x2), x3^2
    - Interaction: x1 * x5
    """
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    return (
        2.0 * x1  # Linear main effect
        + 3.0 * jnp.sin(2 * jnp.pi * x2)  # Nonlinear main effect
        + 1.5 * (x3 - 0.5) ** 2  # Quadratic main effect
        + 1.0 * x4  # Linear main effect
        + 2.5 * x1 * x5  # Interaction effect
    )


# Generate data
key, subkey = jr.split(key)
n_synthetic = 200
X_synthetic = jr.uniform(subkey, (n_synthetic, 5), minval=0, maxval=1)
y_synthetic = complex_additive_function(X_synthetic).reshape(-1, 1)

# Add noise
noise_key, key = jr.split(key)
noise_synthetic = 0.2 * jr.normal(noise_key, y_synthetic.shape)
y_synthetic = y_synthetic + noise_synthetic

# Train/test split
n_train_synthetic = int(0.8 * n_synthetic)
indices = jr.permutation(key, n_synthetic)

X_train_synthetic = X_synthetic[indices[:n_train_synthetic]]
y_train_synthetic = y_synthetic[indices[:n_train_synthetic]]
X_test_synthetic = X_synthetic[indices[n_train_synthetic:]]
y_test_synthetic = y_synthetic[indices[n_train_synthetic:]]

# Create dataset
synthetic_dataset = gpx.Dataset(X=X_train_synthetic, y=y_train_synthetic)
feature_names = ["x1 (linear)", "x2 (sin)", "x3 (quad)", "x4 (linear)", "x5 (interact)"]

print("Synthetic Additive Function Features:")
for i, name in enumerate(feature_names):
    print(f"  x{i + 1}: {name}")
print(
    f"Synthetic data: {X_train_synthetic.shape[0]} train, {X_test_synthetic.shape[0]} test samples"
)

# %%
# Fit OAK model to synthetic data
synthetic_oak_kernel = gpx.kernels.OrthogonalAdditiveKernel(
    lengthscales=[0.2] * 5,
    max_order=2,
    input_means=[0.5] * 5,
    input_scales=[0.3] * 5,
)

synthetic_oak_prior = gpx.gps.Prior(
    kernel=synthetic_oak_kernel, mean_function=mean_function
)
synthetic_oak_posterior = synthetic_oak_prior * gpx.likelihoods.Gaussian(
    num_datapoints=len(y_train_synthetic)
)

# Optimize
print("Optimizing OAK on synthetic additive data...")
synthetic_opt_posterior, _ = gpx.fit(
    model=synthetic_oak_posterior,
    objective=loss_fn,
    train_data=synthetic_dataset,
    optim=ox.adam(learning_rate=5e-3),
    num_iters=300,
    key=key,
)

# Make predictions
synthetic_pred = synthetic_opt_posterior.predict(X_test_synthetic, synthetic_dataset)
synthetic_pred_dist = synthetic_opt_posterior.likelihood(synthetic_pred)
synthetic_pred_mean = synthetic_pred_dist.mean.squeeze()
synthetic_rmse = jnp.sqrt(
    jnp.mean((synthetic_pred_mean - y_test_synthetic.squeeze()) ** 2)
)

# Compute feature importance
synthetic_sobol = gpx.kernels.compute_sobol_indices(
    synthetic_opt_posterior, synthetic_dataset.X, synthetic_dataset.y
)

print(f"Synthetic data OAK RMSE: {synthetic_rmse:.3f}")
print("Feature importance (Sobol indices):")
for component_name, value in synthetic_sobol.items():
    print(f"  {component_name}: {value:.3f}")

# %% [markdown]
# ## Final Visualization: Synthetic Additive Function Results

# %%
# Create final comparison visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Predictions vs true values
axes[0].scatter(
    y_test_synthetic.squeeze(), synthetic_pred_mean, alpha=0.7, color=cols[0]
)
axes[0].plot(
    [y_test_synthetic.min(), y_test_synthetic.max()],
    [y_test_synthetic.min(), y_test_synthetic.max()],
    "k--",
    alpha=0.8,
)
axes[0].set_xlabel("True Function Values")
axes[0].set_ylabel("Predicted Function Values")
axes[0].set_title(f"Synthetic Function Predictions\n(RMSE={synthetic_rmse:.3f})")

# Feature importance
synthetic_indices_values = list(synthetic_sobol.values())
synthetic_indices_names = [
    name.replace("_", " ").title() for name in synthetic_sobol.keys()
]

axes[1].bar(
    synthetic_indices_names,
    synthetic_indices_values,
    color=cols[: len(synthetic_indices_values)],
    alpha=0.7,
)
axes[1].set_ylabel("Sobol Index")
axes[1].set_title("Feature Importance")
axes[1].tick_params(axis="x", rotation=45)

# Feature effects visualization
feature_effects = []
for i in range(5):
    if i == 0:  # x1: linear + interaction
        effect = 2.0 + 2.5 * 0.5  # linear + average interaction
    elif i == 1:  # x2: sin effect
        effect = 3.0  # amplitude of sine
    elif i == 2:  # x3: quadratic
        effect = 1.5 * 0.25  # maximum quadratic effect
    elif i == 3:  # x4: linear
        effect = 1.0
    else:  # x5: interaction only
        effect = 2.5 * 0.5  # average interaction effect
    feature_effects.append(abs(effect))

axes[2].bar([f"x{i + 1}" for i in range(5)], feature_effects, color=cols[:5], alpha=0.7)
axes[2].set_ylabel("True Effect Magnitude")
axes[2].set_title("Known Feature Effects")
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()

# %% [markdown]
# ## Summary and Key Takeaways
#
# The Orthogonal Additive Kernel (OAK) offers several advantages for interpretable machine learning:
#
# ### ✅ **Key Benefits:**
# 1. **Interpretability**: Provides automatic feature importance via Sobol indices
# 2. **Additive Structure**: Decomposes functions into identifiable additive components
# 3. **Efficiency**: Newton-Girard algorithm scales polynomially, not exponentially
# 4. **Orthogonality**: Components are orthogonal, ensuring unique contributions
# 5. **Flexibility**: Supports arbitrary interaction orders via `max_order` parameter
#
# ### 🎯 **When to Use OAK:**
# - When model interpretability is crucial
# - For additive or approximately additive functions
# - When you need feature importance analysis
# - For scientific applications requiring component identification
#
# ### ⚙️ **Key Parameters:**
# - `lengthscales`: Controls smoothness in each dimension
# - `max_order`: Maximum interaction order (computational vs expressivity tradeoff)
# - `input_means`/`input_scales`: Gaussian input density parameters
# - `variances`: Prior importance of each interaction order
#
# ### 📊 **Model Diagnostics:**
# - Use Sobol indices to understand feature/interaction importance
# - Monitor variance parameters to see which interaction orders matter
# - Compare with standard GP kernels for performance validation
#
# The OAK kernel represents a powerful tool for interpretable GP modeling, bridging the gap
# between model expressivity and interpretability in Gaussian process regression.

# %% [markdown]
# ## Further Reading
#
# - **Original Paper**: Lu et al. (2022). "Scalable Gaussian Process Regression and Variable Selection using Orthogonal Additive Kernel." https://arxiv.org/abs/2206.09861
# - **GPJax Documentation**: https://docs.jaxgaussianprocesses.com/
# - **Sobol Sensitivity Analysis**: Sobol, I. M. (2001). "Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates."
# - **ANOVA Decomposition**: Hoeffding, W. (1948). "A class of statistics with asymptotically normal distribution."

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'GPJax Team'
