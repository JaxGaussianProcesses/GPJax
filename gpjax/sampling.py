from typing import Callable

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd

from .gps import ConjugatePosterior, NonConjugatePosterior, Prior, SpectralPosterior
from .kernels import cross_covariance, gram
from .likelihoods import predictive_moments
from .predict import mean, variance
from .types import Array, Dataset
from .utils import I, concat_dictionaries


@dispatch(Prior, dict, Dataset)
def random_variable(
    gp: Prior, params: dict, sample_points: Dataset, jitter_amount: float = 1e-6
) -> tfd.Distribution:
    X = sample_points.X
    N = sample_points.n
    mu = gp.mean_function(X)
    gram_matrix = gram(gp.kernel, X, params)
    jitter_matrix = I(N) * jitter_amount
    covariance = gram_matrix + jitter_matrix
    return tfd.MultivariateNormalFullCovariance(mu.squeeze(), covariance)


@dispatch(ConjugatePosterior, dict, Dataset)
def random_variable(
    gp: ConjugatePosterior,
    params: dict,
    training: Dataset,
    jitter_amount: float = 1e-6,
) -> Callable:
    # TODO: Return kernel matrices here to avoid replicated computation.
    meanf = mean(gp, params, training)
    covf = variance(gp, params, training)

    def build_rv(test_points: Array):
        n = test_points.shape[0]
        mu = meanf(test_points)
        cov = covf(test_points)
        return tfd.MultivariateNormalFullCovariance(
            mu.squeeze(), cov + I(n) * jitter_amount
        )

    return build_rv


@dispatch(NonConjugatePosterior, dict, Dataset)
def random_variable(
    gp: NonConjugatePosterior,
    params: dict,
    training: Dataset,
) -> Callable:
    nu = params["latent"]
    N = training.n
    X, y = training.X, training.y
    Kff = gram(gp.prior.kernel, X, params)
    L = jnp.linalg.cholesky(Kff + jnp.eye(N) * 1e-6)

    def build_rv(test_points: Array):
        Kfx = cross_covariance(gp.prior.kernel, X, test_points, params)
        Kxx = gram(gp.prior.kernel, test_points, params)
        A = solve_triangular(L, Kfx.T, lower=True)
        latent_var = Kxx - jnp.sum(jnp.square(A), -2)
        latent_mean = jnp.matmul(A.T, nu)
        lvar = jnp.diag(latent_var)
        moment_fn = predictive_moments(gp.likelihood)
        return moment_fn(latent_mean.ravel(), lvar)

    return build_rv


@dispatch(SpectralPosterior, dict, Dataset)
def random_variable(
    gp: SpectralPosterior,
    params: dict,
    training: Dataset,
    static_params: dict = None,
) -> tfd.Distribution:
    X, y = training.X, training.y

    params = concat_dictionaries(params, static_params)
    m = gp.prior.kernel.num_basis
    w = params["basis_fns"] / params["lengthscale"]
    phi = gp.prior.kernel._build_phi(X, params)

    A = (params["variance"] / m) * jnp.matmul(jnp.transpose(phi), phi) + params[
        "obs_noise"
    ] * I(2 * m)

    RT = jnp.linalg.cholesky(A)
    R = jnp.transpose(RT)

    RtiPhit = solve_triangular(RT, jnp.transpose(phi))
    Rtiphity = jnp.matmul(RtiPhit, y)

    alpha = params["variance"] / m * solve_triangular(R, Rtiphity, lower=False)

    def build_rv(test_points: Array):
        N = test_points.shape[0]
        phistar = jnp.matmul(test_points, jnp.transpose(w))
        phistar = jnp.hstack([jnp.cos(phistar), jnp.sin(phistar)])
        mean = jnp.matmul(phistar, alpha)

        RtiPhistart = solve_triangular(RT, jnp.transpose(phistar))
        PhiRistar = jnp.transpose(RtiPhistart)
        cov = (
            params["obs_noise"]
            * params["variance"]
            / m
            * jnp.matmul(PhiRistar, jnp.transpose(PhiRistar))
            + I(N) * 1e-6
        )
        return tfd.MultivariateNormalFullCovariance(mean.squeeze(), cov)

    return build_rv


@dispatch(jnp.DeviceArray, Prior, dict, Dataset)
def sample(
    key: jnp.DeviceArray, gp: Prior, params: dict, training: Dataset, n_samples: int = 1
) -> Array:
    rv = random_variable(gp, params, training)
    return rv.sample(sample_shape=(n_samples,), seed=key)


@dispatch(jnp.DeviceArray, Prior, dict, Array)
def sample(
    key: jnp.DeviceArray, gp: Prior, params: dict, X: Array, n_samples: int = 1
) -> Array:
    training = Dataset(X=X, y=jnp.ones_like(X))
    rv = random_variable(gp, params, training)
    return rv.sample(sample_shape=(n_samples,), seed=key)


@dispatch(jnp.DeviceArray, tfd.Distribution)
def sample(
    key: jnp.DeviceArray, random_variable: tfd.Distribution, n_samples: int = 1
) -> Array:
    return random_variable.sample(sample_shape=(n_samples,), seed=key)
