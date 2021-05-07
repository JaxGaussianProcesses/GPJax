from gpjax.parameters import initialise, build_all_transforms
from gpjax.objectives import marginal_ll
from gpjax.sampling import sample
from jax import jit, value_and_grad
import jax.numpy as jnp
from jax.experimental import optimizers
import matplotlib.pyplot as plt


def fit(posterior, nits, data, configs):
    params = initialise(posterior)
    constrainer, unconstrainer = build_all_transforms(params.keys(), configs)

    mll = jit(marginal_ll(posterior, transform=constrainer, negative=True))

    opt_init, opt_update, get_params = optimizers.adam(step_size=0.05)
    opt_state = opt_init(params)

    def step(i, opt_state):
        p = get_params(opt_state)
        v, g = value_and_grad(mll)(p, data)
        return opt_update(i, g, opt_state), v

    for i in range(nits):
        opt_state, mll_estimate = step(i, opt_state)
    print(f"{posterior.prior.kernel.name} GP's marginal log-likelihood: {mll_estimate: .2f}")

    final_params = constrainer(get_params(opt_state))
    return final_params


def plot(key, rv, query_points, training_data, ax, n_samples=100):
    mu = rv.mean()
    sigma = rv.variance()
    one_stddev = 1.96 * jnp.sqrt(sigma)
    posterior_samples = sample(key, rv, n_samples=n_samples)
    plt.grid(color="#888888")  # Color the grid
    ax.spines["polar"].set_visible(False)  # Show or hide the plot spine
    ax.plot(query_points, posterior_samples.T, color="tab:blue", alpha=0.1)
    ax.fill_between(
        query_points.squeeze(),
        mu - one_stddev,
        mu + one_stddev,
        color="tab:blue",
        alpha=0.2,
        label=r"1 Posterior s.d.",
    )
    ax.plot(query_points.squeeze(), mu - one_stddev, linestyle="--", color="tab:blue")
    ax.plot(query_points.squeeze(), mu + one_stddev, linestyle="--", color="tab:blue")
    ax.plot(query_points, mu, color="#b5121b", label='Posterior mean')
    ax.scatter(training_data.X, training_data.y, color="tab:orange", alpha=1, label="observations")
    ax.legend()
