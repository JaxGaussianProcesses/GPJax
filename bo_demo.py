from jax.config import config

config.update("jax_enable_x64", True)


import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt

import gpjax as gpx
from gpjax.bayes_opt.acquisition_functions import (
    ThompsonSamplingAcquisitionFunctionBuilder,
)
from gpjax.bayes_opt.acquisition_optimiser import ContinuousAcquisitionOptimiser
from gpjax.bayes_opt.bayesian_optimiser import BayesianOptimiser
from gpjax.bayes_opt.function_evaluator import (
    OBJECTIVE,
    build_function_evaluator,
)
from gpjax.bayes_opt.posterior_optimiser import AdamPosteriorOptimiser
from gpjax.bayes_opt.search_space import BoxSearchSpace
from gpjax.typing import (
    Array,
    Float,
)

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


def forrester(x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
    return (6 * x - 2) ** 2 * jnp.sin(12 * x - 4)


def print_most_recent_query(
    bo: BayesianOptimiser, last_queried_point: Float[Array, "1 D"]
):
    print(f"Most recently queried point: {last_queried_point}")


def plot_bo_iteration(bo: BayesianOptimiser, last_queried_point: Float[Array, "1 D"]):
    posterior = bo.posteriors[OBJECTIVE]
    dataset = bo.datasets[OBJECTIVE]
    plt_x = jnp.linspace(0, 1, 1000).reshape(-1, 1)
    forrester_y = forrester(plt_x)
    acquisition_fn = bo.acquisition_function_builder.build_acquisition_function(
        bo.posteriors, bo.datasets, bo.key
    )
    sample_y = -acquisition_fn(plt_x)

    latent_dist = posterior.predict(plt_x, train_data=dataset)
    predictive_dist = posterior.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()

    fig, ax = plt.subplots()
    ax.plot(plt_x, predictive_mean, label="Predictive Mean", color=cols[1])
    ax.fill_between(
        plt_x.squeeze(),
        predictive_mean - 2 * predictive_std,
        predictive_mean + 2 * predictive_std,
        alpha=0.2,
        label="Two sigma",
        color=cols[1],
    )
    ax.plot(
        plt_x,
        predictive_mean - 2 * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    ax.plot(
        plt_x,
        predictive_mean + 2 * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    ax.plot(plt_x, sample_y, label="Posterior Sample")
    ax.plot(
        plt_x,
        forrester_y,
        label="Forrester Function",
        color=cols[0],
        linestyle="--",
        linewidth=2,
    )
    ax.axvline(x=0.757, linestyle=":", color=cols[3], label="True Optimum")
    ax.scatter(dataset.X, dataset.y, label="Observations", color=cols[2], zorder=2)
    ax.scatter(
        last_queried_point,
        -acquisition_fn(last_queried_point),
        label="Posterior Sample Optimum",
        marker="*",
        color=cols[3],
        zorder=3,
    )
    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
    plt.show()


if __name__ == "__main__":
    key = jr.PRNGKey(42)
    function_evaluator = build_function_evaluator({OBJECTIVE: forrester})
    lower_bounds = jnp.array([0.0])
    upper_bounds = jnp.array([1.0])
    print(f"Lower Bounds Shape: {lower_bounds.shape}")
    search_space = BoxSearchSpace(lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    initial_x = search_space.sample_halton(5, key)
    print(f"Initial x: {initial_x}")
    initial_datasets = function_evaluator(initial_x)
    print(f"Initial Datasets: {initial_datasets}")

    mean = gpx.mean_functions.Zero()
    kernel = gpx.kernels.Matern52()
    prior = gpx.Prior(mean_function=mean, kernel=kernel)
    likelihood = gpx.Gaussian(num_datapoints=5, obs_noise=jnp.array(1e-6))
    likelihood = likelihood.replace_trainable(obs_noise=False)
    posterior = prior * likelihood
    posteriors = {OBJECTIVE: posterior}
    posterior_optimiser = AdamPosteriorOptimiser(
        num_iters=1000,
        learning_rate=0.01,
        objective=gpx.objectives.ConjugateMLL(negative=True),
    )
    acquisition_function_builder = ThompsonSamplingAcquisitionFunctionBuilder(
        num_rff_features=500
    )
    acquisition_optimiser = ContinuousAcquisitionOptimiser(num_initial_samples=100)
    bo = BayesianOptimiser(
        search_space=search_space,
        posteriors=posteriors,
        datasets=initial_datasets,
        posterior_optimiser=posterior_optimiser,
        acquisition_function_builder=acquisition_function_builder,
        acquisition_optimiser=acquisition_optimiser,
        key=key,
        black_box_function_evaluator=function_evaluator,
        post_ask=[print_most_recent_query, plot_bo_iteration],
    )

    results = bo.run(10)
    print(f"Queried x values: {results[OBJECTIVE].X}")
    print(f"Queried y values: {results[OBJECTIVE].y}")
