import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd

from .gps import ConjugatePosterior, NonConjugatePosterior, Prior
from .kernels import cross_covariance, gram
from .likelihoods import predictive_moments
from .predict import mean, variance
from .types import Array
from .utils import I


@dispatch(Prior, dict, jnp.DeviceArray)
def random_variable(
    gp: Prior, params: dict, sample_points: Array, jitter_amount: float = 1e-6
) -> tfd.Distribution:
    mu = gp.mean_function(sample_points)
    gram_matrix = params["variance"] * gram(gp.kernel, sample_points / params["lengthscale"])
    jitter_matrix = I(sample_points.shape[0]) * jitter_amount
    covariance = gram_matrix + jitter_matrix
    return tfd.MultivariateNormalFullCovariance(mu.squeeze(), covariance)


@dispatch(ConjugatePosterior, dict, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray)
def random_variable(
    gp: ConjugatePosterior,
    params: dict,
    sample_points: Array,
    train_inputs: Array,
    train_outputs: Array,
    jitter_amount: float = 1e-6,
) -> tfd.Distribution:
    n = sample_points.shape[0]
    # TODO: Return kernel matrices here to avoid replicated computation.
    mu = mean(gp, params, sample_points, train_inputs, train_outputs)
    cov = variance(gp, params, sample_points, train_inputs, train_outputs)
    return tfd.MultivariateNormalFullCovariance(mu.squeeze(), cov + I(n) * jitter_amount)


@dispatch(NonConjugatePosterior, dict, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray)
def random_variable(
    gp: NonConjugatePosterior,
    params: dict,
    sample_points: Array,
    train_inputs: Array,
    train_outputs: Array,
) -> tfd.Distribution:
    ell, alpha, nu = params["lengthscale"], params["variance"], params["latent"]
    n_train = train_inputs.shape[0]
    Kff = alpha * gram(gp.prior.kernel, train_inputs / ell)
    Kfx = alpha * cross_covariance(gp.prior.kernel, train_inputs / ell, sample_points / ell)
    Kxx = alpha * gram(gp.prior.kernel, sample_points / ell)
    L = jnp.linalg.cholesky(Kff + jnp.eye(train_inputs.shape[0]) * 1e-6)

    A = solve_triangular(L, Kfx.T, lower=True)
    latent_var = Kxx - jnp.sum(jnp.square(A), -2)
    latent_mean = jnp.matmul(A.T, nu)
    lvar = jnp.diag(latent_var)
    moment_fn = predictive_moments(gp.likelihood)
    return moment_fn(latent_mean.ravel(), lvar)


@dispatch(jnp.DeviceArray, Prior, dict, jnp.DeviceArray)
def sample(key, gp, params, sample_points, n_samples=1) -> Array:
    rv = random_variable(gp, params, sample_points)
    return rv.sample(sample_shape=(n_samples,), seed=key)


@dispatch(jnp.DeviceArray, tfd.Distribution)
def sample(key: jnp.DeviceArray, random_variable: tfd.Distribution, n_samples: int = 1) -> Array:
    return random_variable.sample(sample_shape=(n_samples,), seed=key)
