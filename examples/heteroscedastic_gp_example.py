"""
Example of using the heteroscedastic Gaussian likelihood in GPJax.

This example demonstrates how to create and fit a heteroscedastic Gaussian process
model using GPJax. The model consists of two Gaussian processes:
1. A signal GP (f) that models the mean function
2. A noise GP (g) that models the log of the observation noise variance

The observation model is:
    y(x) = f(x) + ε(x), where ε(x) ~ N(0, exp(g(x)))
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
import numpy as np
from flax import nnx

import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.objectives import MeanFieldElboObjective, NegativeMeanFieldElboObjective
from gpjax.variational_families import MeanFieldGaussian


def generate_heteroscedastic_data(key, n_points=50):
    """Generate synthetic data with heteroscedastic noise."""
    # Set up input points
    X = jnp.linspace(0, 10, n_points).reshape(-1, 1)

    # True signal function: sin(x)
    true_signal = jnp.sin(X.flatten())

    # Heteroscedastic noise: noise increases with x
    noise_std = 0.1 + 0.1 * X.flatten()
    noise = jr.normal(key, shape=(n_points,)) * noise_std

    # Observed targets
    y = true_signal + noise

    return X, y, true_signal, noise_std


def main():
    # Set random seed for reproducibility
    key = jr.PRNGKey(42)

    # Generate synthetic data
    X_train, y_train, true_signal, true_noise_std = generate_heteroscedastic_data(key)

    # Create dataset
    dataset = Dataset(X=X_train, y=y_train.reshape(-1, 1))

    # Create signal GP prior
    signal_mean = gpx.mean_functions.Zero()
    signal_kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
    signal_prior = gpx.gps.Prior(mean_function=signal_mean, kernel=signal_kernel)

    # Create noise GP prior (for modeling log variance)
    # Initialize with a constant mean function at log(0.1²) = -2.3
    noise_mean = gpx.mean_functions.Constant(constant=-2.3)
    noise_kernel = gpx.kernels.RBF(lengthscale=2.0, variance=0.5)
    noise_prior = gpx.gps.Prior(mean_function=noise_mean, kernel=noise_kernel)

    # Create heteroscedastic Gaussian likelihood
    num_datapoints = X_train.shape[0]
    signal_prior, likelihood = gpx.create_heteroscedastic_gp(
        signal_prior=signal_prior,
        noise_prior=noise_prior,
        num_datapoints=num_datapoints,
        clip_min=-10.0,
        clip_max=10.0,
    )

    # Create variational posterior for the signal GP
    signal_posterior = gpx.gps.NonConjugatePosterior(
        prior=signal_prior,
        likelihood=likelihood,
        latent=None,  # Will be initialized automatically
    )

    # Create variational distribution for both signal and noise GPs
    # We need to model both f and g jointly
    variational_family = MeanFieldGaussian(
        posterior=signal_posterior,
        num_datapoints=num_datapoints,
        num_latent_gps=2,  # One for signal, one for noise
    )

    # Create objective function (negative ELBO)
    objective = NegativeMeanFieldElboObjective(
        posterior=signal_posterior,
        likelihood=likelihood,
        variational_family=variational_family,
    )

    # Set up optimizer
    learning_rate = 0.01
    optimizer = optax.adam(learning_rate=learning_rate)

    # Initialize parameters
    params = nnx.state(
        signal_posterior=signal_posterior,
        variational_family=variational_family,
    )

    # Fit the model
    params, history = gpx.fit(
        objective=objective,
        params=params,
        data=dataset,
        optax_optimizer=optimizer,
        num_iters=1000,
        key=key,
    )

    # Create test points for prediction
    X_test = jnp.linspace(0, 10, 200).reshape(-1, 1)

    # Extract the trained models
    signal_posterior = params["signal_posterior"]
    variational_family = params["variational_family"]

    # Get the variational parameters
    q_mu, q_sqrt = variational_family.params

    # Split the variational parameters for signal and noise
    q_mu_signal = q_mu[:num_datapoints]
    q_mu_noise = q_mu[num_datapoints:]

    # Make predictions for the signal GP
    signal_dist = signal_posterior.prior.predict(X_test)

    # Make predictions for the noise GP
    noise_dist = noise_prior.predict(X_test)

    # Get the predictive distribution with heteroscedastic noise
    predictive_dist = likelihood.predict(signal_dist, noise_dist)

    # Extract mean and standard deviation
    pred_mean = predictive_dist.mean()
    pred_stddev = predictive_dist.stddev()

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot training data
    plt.subplot(2, 1, 1)
    plt.scatter(X_train, y_train, color="black", alpha=0.5, label="Training data")
    plt.plot(X_test, jnp.sin(X_test), "r--", label="True signal (sin(x))")

    # Plot predictive mean and confidence intervals
    plt.plot(X_test, pred_mean, "b-", label="Predictive mean")
    plt.fill_between(
        X_test.flatten(),
        pred_mean - 2 * pred_stddev,
        pred_mean + 2 * pred_stddev,
        color="blue",
        alpha=0.2,
        label="95% confidence interval",
    )

    plt.title("Heteroscedastic GP Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # Plot the learned noise standard deviation
    plt.subplot(2, 1, 2)
    plt.plot(X_train, true_noise_std, "r--", label="True noise std")
    plt.plot(X_test, pred_stddev, "b-", label="Learned noise std")
    plt.title("Noise Standard Deviation")
    plt.xlabel("x")
    plt.ylabel("Noise std")
    plt.legend()

    plt.tight_layout()
    plt.savefig("heteroscedastic_gp_results.png")
    plt.show()

    # Plot the training history
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Negative ELBO")
    plt.yscale("log")
    plt.savefig("heteroscedastic_gp_training.png")
    plt.show()


if __name__ == "__main__":
    main()
