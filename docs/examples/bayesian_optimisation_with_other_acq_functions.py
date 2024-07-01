# %% [markdown]
# # Bayesian Optimisation beyond Thompson Sampling
#
# In [a previous guide](), we gave an introduction to Bayesian optimisation:
# a framework for optimising black-box function that leverages the
# uncertainty estimates that come from Gaussian processes.

# %%
# Enable Float64 for more stable matrix inversions.
from jax import config

config.update("jax_enable_x64", True)

import jax
from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook, Float, Int
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import optax as ox
import tensorflow_probability.substrates.jax as tfp
from typing import List, Tuple

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
from gpjax.typing import Array, FunctionalSample, ScalarFloat
from jaxopt import ScipyBoundedMinimize

key = jr.key(1337)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]

# In a few words, Bayesian optimisation starts by fitting a Gaussian
# process to the data we have collected so far about the objective function,
# and then uses this model to construct an **acquisition function**
# which tells us which parts of the input domain have the potential
# of improving our optimization. Unlike the black-box objective, the
# acquisition function is easy to query and optimize.

# Acquisition functions come in many flavors. In the previous guide,
# we sampled from the Gaussian Process' predictive posterior and
# optimized said sample. This is known as Thompson Sampling,
# and it is a popular acquisition function due to its simplicity
# and ease of parallelization.

# In this guide we will introduce a new acquisition function:
# probability of improvement [TODO:ADDCITE](). This acquisition function can be formally
# defined as

# $$ \text{PI}(x) = \text{Prob}(f(x) < f(x_{\text{best}})) $$

# where $f(x)$ is the objective functionw we aim at **minimizing**,
# and $x_{\text{best}}$ is the best point we have seen so far (i.e.
# the point with the lowest value of $f$). The name is clear: it measures
# the probability that a given point $x$ will **improve** our
# optimization trace.

# %% [markdown]
# ## Optimizing a 1D function: Forrester
#
# Just like in our previous guide, let's start by defining the
# [Forrester objective function](https://www.sfu.ca/~ssurjano/forretal08.html).


# %%
def standardised_forrester(x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
    mean = 0.45321
    std = 4.4258
    return ((6 * x - 2) ** 2 * jnp.sin(12 * x - 4) - mean) / std


# %%
lower_bound = jnp.array([0.0])
upper_bound = jnp.array([1.0])
initial_sample_num = 5

initial_x = tfp.mcmc.sample_halton_sequence(
    dim=1, num_results=initial_sample_num, seed=key, dtype=jnp.float64
).reshape(-1, 1)
initial_y = standardised_forrester(initial_x)
D = gpx.Dataset(X=initial_x, y=initial_y)

# %% [markdown]

# ...defining the Gaussian Process model...


# %%
def return_optimised_posterior(
    data: gpx.Dataset, prior: gpx.base.Module, key: Array
) -> gpx.base.Module:
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=data.n, obs_stddev=jnp.array(1e-6)
    )  # Our function is noise-free, so we set the observation noise's standard deviation to a very small value
    likelihood = likelihood.replace_trainable(obs_stddev=False)

    posterior = prior * likelihood

    negative_mll = gpx.objectives.ConjugateMLL(negative=True)
    negative_mll(posterior, train_data=data)
    negative_mll = jit(negative_mll)

    opt_posterior, _ = gpx.fit(
        model=posterior,
        objective=negative_mll,
        train_data=data,
        optim=ox.adam(learning_rate=0.01),
        num_iters=1000,
        safe=True,
        key=key,
        verbose=False,
    )

    return opt_posterior


mean = gpx.mean_functions.Zero()
kernel = gpx.kernels.Matern52()
prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
opt_posterior = return_optimised_posterior(D, prior, key)


from gpjax.decision_making.utility_functions.probability_of_improvement import (
    ProbabilityOfImprovement,
)

utility_function_builder = ProbabilityOfImprovement()
utility_function = utility_function_builder.build_utility_function(
    posteriors={"OBJECTIVE": opt_posterior}, datasets={"OBJECTIVE": D}, key=key
)

from gpjax.decision_making.utility_maximizer import (
    ContinuousSinglePointUtilityMaximizer,
)
from gpjax.decision_making.utility_functions.base import SinglePointUtilityFunction
from gpjax.decision_making.search_space import ContinuousSearchSpace


def optimize_acquisition_function(
    utility_function: SinglePointUtilityFunction,
    key: Array,
    lower_bounds: Float[Array, "D"],
    upper_bounds: Float[Array, "D"],
    num_initial_samples: int = 100,
    num_restarts: int = 5,
):
    optimizer = ContinuousSinglePointUtilityMaximizer(
        num_initial_samples=num_initial_samples, num_restarts=num_restarts
    )

    search_space = ContinuousSearchSpace(
        lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )

    x_next_best = optimizer.maximize(
        utility_function, search_space=search_space, key=key
    )

    return x_next_best


# %%
def construct_acquisition_function(
    opt_posterior: gpx.base.Module,
    dataset: gpx.Dataset,
    key: Array,
) -> SinglePointUtilityFunction:
    utility_function_builder = ProbabilityOfImprovement()
    utility_function = utility_function_builder.build_utility_function(
        posteriors={"OBJECTIVE": opt_posterior},
        datasets={"OBJECTIVE": dataset},
        key=key,
    )

    return utility_function


def propose_next_candidate(
    utility_function: SinglePointUtilityFunction,
    key: Array,
    lower_bounds: Float[Array, "D"],
    upper_bounds: Float[Array, "D"],
) -> Float[Array, "D 1"]:
    queried_x = optimize_acquisition_function(
        utility_function, key, lower_bounds, upper_bounds, num_initial_samples=100
    )

    return queried_x


def run_one_bo_loop_in_1D(
    objective_function: standardised_forrester,
    opt_posterior: gpx.base.Module,
    dataset: gpx.Dataset,
    key: Array,
    plot: bool = True,
) -> Float[Array, "D 1"]:
    domain = jnp.linspace(0, 1, 1000).reshape(-1, 1)
    objective_values = objective_function(domain)

    latent_distribution = opt_posterior.predict(domain, train_data=dataset)
    predictive_distribution = opt_posterior.likelihood(latent_distribution)

    predictive_mean = predictive_distribution.mean()
    predictive_std = predictive_distribution.stddev()

    # Building PI
    utility_function = construct_acquisition_function(opt_posterior, dataset, key)
    utility_function_values = utility_function(domain)

    # Optimizing the acq. function
    lower_bound = jnp.array([0.0])
    upper_bound = jnp.array([1.0])
    queried_x = propose_next_candidate(utility_function, key, lower_bound, upper_bound)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(domain, predictive_mean, label="Predictive Mean", color=cols[1])
        ax.fill_between(
            domain.squeeze(),
            predictive_mean - 2 * predictive_std,
            predictive_mean + 2 * predictive_std,
            alpha=0.2,
            label="Two sigma",
            color=cols[1],
        )
        ax.plot(
            domain,
            predictive_mean - 2 * predictive_std,
            linestyle="--",
            linewidth=1,
            color=cols[1],
        )
        ax.plot(
            domain,
            predictive_mean + 2 * predictive_std,
            linestyle="--",
            linewidth=1,
            color=cols[1],
        )
        ax.plot(domain, utility_function_values, label="Probability of Improvement")
        ax.plot(
            domain,
            objective_values,
            label="Forrester Function",
            color=cols[0],
            linestyle="--",
            linewidth=2,
        )
        ax.axvline(x=0.757, linestyle=":", color=cols[3], label="True Optimum")
        ax.scatter(dataset.X, dataset.y, label="Observations", color=cols[2], zorder=2)
        ax.scatter(
            queried_x,
            utility_function(queried_x),
            label="Probability of Improvement Optimum",
            marker="*",
            color=cols[3],
            zorder=3,
        )
        ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))
        plt.show()

    return queried_x


# plot_bayes_opt_using_pi(standardised_forrester, opt_posterior, D, key)

# %%
bo_iters = 5

# Set up initial dataset
initial_x = tfp.mcmc.sample_halton_sequence(
    dim=1, num_results=initial_sample_num, seed=key, dtype=jnp.float64
).reshape(-1, 1)
initial_y = standardised_forrester(initial_x)
D = gpx.Dataset(X=initial_x, y=initial_y)

for i in range(bo_iters):
    key, subkey = jr.split(key)

    # Generate optimised posterior using previously observed data
    mean = gpx.mean_functions.Zero()
    kernel = gpx.kernels.Matern52()
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
    opt_posterior = return_optimised_posterior(D, prior, subkey)

    queried_x = run_one_bo_loop_in_1D(standardised_forrester, opt_posterior, D, key)

    # Evaluate the black-box function at the best point observed so far, and add it to the dataset
    y_star = standardised_forrester(queried_x)
    print(f"Queried Point: {queried_x}, Black-Box Function Value: {y_star}")
    D = D + gpx.Dataset(X=queried_x, y=y_star)


# %%
def standardised_six_hump_camel(x: Float[Array, "N 2"]) -> Float[Array, "N 1"]:
    mean = 1.12767
    std = 1.17500
    x1 = x[..., :1]
    x2 = x[..., 1:]
    term1 = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return (term1 + term2 + term3 - mean) / std


# %%
x1 = jnp.linspace(-2, 2, 100)
x2 = jnp.linspace(-1, 1, 100)
x1, x2 = jnp.meshgrid(x1, x2)
x = jnp.stack([x1.flatten(), x2.flatten()], axis=1)
y = standardised_six_hump_camel(x)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(
    x1,
    x2,
    y.reshape(x1.shape[0], x2.shape[0]),
    linewidth=0,
    cmap=cm.coolwarm,
    antialiased=False,
)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()


# %%
x_star_one = jnp.array([[0.0898, -0.7126]])
x_star_two = jnp.array([[-0.0898, 0.7126]])
fig, ax = plt.subplots()
contour_plot = ax.contourf(
    x1, x2, y.reshape(x1.shape[0], x2.shape[0]), cmap=cm.coolwarm, levels=40
)
ax.scatter(
    x_star_one[0][0], x_star_one[0][1], marker="*", color=cols[2], label="Global Minima"
)
ax.scatter(x_star_two[0][0], x_star_two[0][1], marker="*", color=cols[2])
ax.set_xlabel("x1")
ax.set_ylabel("x2")
fig.colorbar(contour_plot)
ax.legend()
plt.show()

# %%
lower_bound = jnp.array([-2.0, -1.0])
upper_bound = jnp.array([2.0, 1.0])
initial_sample_num = 5
bo_iters = 20
num_experiments = 5
bo_experiment_results = []

for experiment in range(num_experiments):
    print(f"Starting Experiment: {experiment + 1}")
    # Set up initial dataset
    initial_x = tfp.mcmc.sample_halton_sequence(
        dim=2, num_results=initial_sample_num, seed=key, dtype=jnp.float64
    )
    initial_x = jnp.array(lower_bound + (upper_bound - lower_bound) * initial_x)
    initial_y = standardised_six_hump_camel(initial_x)
    D = gpx.Dataset(X=initial_x, y=initial_y)

    for i in range(bo_iters):
        key, subkey = jr.split(key)

        # Generate optimised posterior
        mean = gpx.mean_functions.Zero()
        kernel = gpx.kernels.Matern52(
            active_dims=[0, 1], lengthscale=jnp.array([1.0, 1.0]), variance=2.0
        )
        prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
        opt_posterior = return_optimised_posterior(D, prior, subkey)

        # Constructing the acq. function
        utility_function = construct_acquisition_function(opt_posterior, D, key)

        # Draw a sample from the posterior, and find the minimiser of it
        queried_x = propose_next_candidate(
            utility_function, key, lower_bound, upper_bound
        )

        # Evaluate the black-box function at the best point observed so far, and add it to the dataset
        y_star = standardised_six_hump_camel(queried_x)
        print(
            f"BO Iteration: {i + 1}, Queried Point: {queried_x}, Black-Box Function Value:"
            f" {y_star}"
        )
        D = D + gpx.Dataset(X=queried_x, y=y_star)
    bo_experiment_results.append(D)

# %%
random_experiment_results = []
for i in range(num_experiments):
    key, subkey = jr.split(key)
    initial_x = bo_experiment_results[i].X[:5]
    initial_y = bo_experiment_results[i].y[:5]
    final_x = jr.uniform(
        key,
        shape=(bo_iters, 2),
        dtype=jnp.float64,
        minval=lower_bound,
        maxval=upper_bound,
    )
    final_y = standardised_six_hump_camel(final_x)
    random_x = jnp.concatenate([initial_x, final_x], axis=0)
    random_y = jnp.concatenate([initial_y, final_y], axis=0)
    random_experiment_results.append(gpx.Dataset(X=random_x, y=random_y))


# %%
def obtain_log_regret_statistics(
    experiment_results: List[gpx.Dataset],
    global_minimum: ScalarFloat,
) -> Tuple[Float[Array, "N 1"], Float[Array, "N 1"]]:
    log_regret_results = []
    for exp_result in experiment_results:
        observations = exp_result.y
        cumulative_best_observations = jax.lax.associative_scan(
            jax.numpy.minimum, observations
        )
        regret = cumulative_best_observations - global_minimum
        log_regret = jnp.log(regret)
        log_regret_results.append(log_regret)

    log_regret_results = jnp.array(log_regret_results)
    log_regret_mean = jnp.mean(log_regret_results, axis=0)
    log_regret_std = jnp.std(log_regret_results, axis=0)
    return log_regret_mean, log_regret_std


bo_log_regret_mean, bo_log_regret_std = obtain_log_regret_statistics(
    bo_experiment_results, -1.8377
)
(
    random_log_regret_mean,
    random_log_regret_std,
) = obtain_log_regret_statistics(random_experiment_results, -1.8377)

# %% [markdown]
# Now, when we plot the mean and standard deviation of the log regret at each iteration,
# we can see that BO outperforms random sampling!

# %%
fig, ax = plt.subplots()
fn_evaluations = jnp.arange(1, bo_iters + initial_sample_num + 1)
ax.plot(fn_evaluations, bo_log_regret_mean, label="Bayesian Optimisation")
ax.fill_between(
    fn_evaluations,
    bo_log_regret_mean[:, 0] - bo_log_regret_std[:, 0],
    bo_log_regret_mean[:, 0] + bo_log_regret_std[:, 0],
    alpha=0.2,
)
ax.plot(fn_evaluations, random_log_regret_mean, label="Random Search")
ax.fill_between(
    fn_evaluations,
    random_log_regret_mean[:, 0] - random_log_regret_std[:, 0],
    random_log_regret_mean[:, 0] + random_log_regret_std[:, 0],
    alpha=0.2,
)
ax.axvline(x=initial_sample_num, linestyle=":")
ax.set_xlabel("Number of Black-Box Function Evaluations")
ax.set_ylabel("Log Regret")
ax.legend()
plt.show()

# %% [markdown]
# It can also be useful to plot the queried points over the course of a single BO run, in
# order to gain some insight into how the algorithm queries the search space. Below
# we do this for one of the BO experiments, and can see that the algorithm initially
# performs some exploration of the search space whilst it is uncertain about the black-box
# function, but it then hones in one one of the global minima of the function, as we would hope!

# %%
fig, ax = plt.subplots()
contour_plot = ax.contourf(
    x1, x2, y.reshape(x1.shape[0], x2.shape[0]), cmap=cm.coolwarm, levels=40
)
ax.scatter(
    x_star_one[0][0],
    x_star_one[0][1],
    marker="*",
    color=cols[2],
    label="Global Minimum",
    zorder=2,
)
ax.scatter(x_star_two[0][0], x_star_two[0][1], marker="*", color=cols[2], zorder=2)
ax.scatter(
    bo_experiment_results[1].X[:, 0],
    bo_experiment_results[1].X[:, 1],
    marker="x",
    color=cols[1],
    label="Bayesian Optimisation Queries",
)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
fig.colorbar(contour_plot)
ax.legend()
plt.show()

# %%
