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
#     display_name: Python 3.9.7 ('gpjax')
#     language: python
#     name: python3
# ---

# %%
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
from jax.config import config
import matplotlib.pyplot as plt
import distrax as dx

key = jr.PRNGKey(123)
config.update("jax_enable_x64", True)

# %%
n_data = 50
n_realisations = 4
noise_limits = (0.3, 0.5)
xlims = (-5, 5)
jitter = 1e-6
n_inducing = 25

true_kernel = gpx.kernels.Matern32()
true_params = gpx.initialise(true_kernel, key)

X = jnp.linspace(*xlims, n_data).reshape(-1, 1)
true_kxx = (
    true_kernel.gram(true_kernel, true_params.params, X) + jnp.eye(n_data) * jitter
)
true_L = true_kxx.triangular_lower()
latent_dist = dx.MultivariateNormalTri(jnp.zeros(n), true_L)
group_y = latent_dist.sample(seed=key, sample_shape=(1,)).T


noise_terms = dx.Uniform(*noise_limits).sample(seed=key, sample_shape=(n_realisations,))

# def add_sig(i):
#     X = jnp.linspace(*xlims, n_data).reshape(-1, 1)
#     group_y = tfp.distributions.MultivariateNormalTriL(np.zeros(n_data), tf.linalg.cholesky(Kxx)).sample(seed=tfp_seed + 10 * i)
#     sample_y = group_y.numpy()
#     return sample_y

realisations = []
individuals_ys = []
datasets = []
for idx, (noise, skey) in enumerate(zip(noise_terms, jr.split(key, n_realisations))):
    # Split the key
    noise_vector = dx.Normal(0, noise).sample(seed=skey, sample_shape=group_y.shape)
    y = group_y + noise_vector
    individuals_ys.append(y)
    realisations.append(gpx.Dataset(X=X, y=y))
    plt.plot(X, y, color="tab:blue")
plt.plot(X, group_y, color="tab:red")

# %%
inducing_points = jnp.linspace(*xlims, n_inducing).reshape(-1, 1)

individual_priors = [
    gpx.Prior(kernel=gpx.kernels.Matern32()) for _ in range(n_realisations)
]
group_prior = gpx.Prior(kernel=gpx.kernels.Matern32())
likelihood = gpx.Gaussian(num_datapoints=n_data)

# %%
import typing as tp
from jaxtyping import Float, Array
from chex import PRNGKey as PRNGKeyType, dataclass
from gpjax.utils import concat_dictionaries


@dataclass
class SHGP(gpx.variational_families.AbstractVariationalFamily):
    individual_priors: tp.List[gpx.Prior]
    group_prior: gpx.Prior
    likelihood: gpx.likelihoods.AbstractLikelihood
    inducing_inputs: Float[Array, "M D"]
    name: str = "Sparse Hierarchical GP"
    diag: tp.Optional[bool] = False

    def _initialise_params(self, key: PRNGKeyType) -> tp.Dict:
        params = {}
        prior_params = [p._initialise_params(key) for p in self.individual_priors]
        params["group"] = self.group_prior._initialise_params(key)
        params["individuals"] = prior_params
        params = concat_dictionaries(
            params,
            {
                "variational_family": {"inducing_inputs": self.inducing_inputs},
                "likelihood": {
                    "obs_noise": self.likelihood._initialise_params(key)["obs_noise"]
                },
            },
        )
        return params

    def predict(self, *args: tp.Any, **kwargs: tp.Any) -> dx.Distribution:
        return super().predict(*args, **kwargs)


shgp = SHGP(
    individual_priors=individual_priors,
    group_prior=group_prior,
    likelihood=likelihood,
    inducing_inputs=inducing_points,
)

# %%
