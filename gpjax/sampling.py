import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd

from .gps import (ConjugatePosterior, NonConjugatePosterior, Prior,
                  SpectralPosterior)
from .kernels import cross_covariance, gram
from .likelihoods import predictive_moments
from .predict import mean, variance
from .types import Array
from .utils import I, concat_dictionaries


@dispatch(Prior, dict, jnp.DeviceArray)
def random_variable(
    gp: Prior, params: dict, sample_points: Array, jitter_amount: float = 1e-6
) -> tfd.Distribution:
    mu = gp.mean_function(sample_points)
    gram_matrix = gram(gp.kernel, sample_points, params)
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
    Kff = gram(gp.prior.kernel, train_inputs, params)
    Kfx = cross_covariance(gp.prior.kernel, train_inputs, sample_points, params)
    Kxx = gram(gp.prior.kernel, sample_points, params)
    L = jnp.linalg.cholesky(Kff + jnp.eye(train_inputs.shape[0]) * 1e-6)

    A = solve_triangular(L, Kfx.T, lower=True)
    latent_var = Kxx - jnp.sum(jnp.square(A), -2)
    latent_mean = jnp.matmul(A.T, nu)
    lvar = jnp.diag(latent_var)
    moment_fn = predictive_moments(gp.likelihood)
    return moment_fn(latent_mean.ravel(), lvar)


@dispatch(SpectralPosterior, dict, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray)
def random_variable(
    gp: SpectralPosterior,
    params: dict,
    train_inputs: Array,
    train_outputs: Array,
    test_inputs: Array,
    static_params: dict = None,
) -> tfd.Distribution:
    params = concat_dictionaries(params, static_params)
    m = gp.prior.kernel.num_basis
    w = params["basis_fns"] / params["lengthscale"]
    phi = gp.prior.kernel._build_phi(train_inputs, params)

    A = (params["variance"] / m) * jnp.matmul(jnp.transpose(phi), phi) + params["obs_noise"] * I(
        2 * m
    )

    RT = jnp.linalg.cholesky(A)
    R = jnp.transpose(RT)

    RtiPhit = solve_triangular(RT, jnp.transpose(phi))
    # Rtiphity=RtiPhit*y_tr;
    Rtiphity = jnp.matmul(RtiPhit, train_outputs)

    alpha = params["variance"] / m * solve_triangular(R, Rtiphity, lower=False)

    phistar = jnp.matmul(test_inputs, jnp.transpose(w))
    # phistar = [cos(phistar) sin(phistar)];                              % test design matrix
    phistar = jnp.hstack([jnp.cos(phistar), jnp.sin(phistar)])
    # out1(beg_chunk:end_chunk) = phistar*alfa;                           % Predictive mean
    mean = jnp.matmul(phistar, alpha)
    print(mean.shape)

    RtiPhistart = solve_triangular(RT, jnp.transpose(phistar))
    PhiRistar = jnp.transpose(RtiPhistart)
    cov = (
        params["obs_noise"]
        * params["variance"]
        / m
        * jnp.matmul(PhiRistar, jnp.transpose(PhiRistar))
        + I(test_inputs.shape[0]) * 1e-6
    )
    return tfd.MultivariateNormalFullCovariance(mean.squeeze(), cov)


@dispatch(jnp.DeviceArray, Prior, dict, jnp.DeviceArray)
def sample(key, gp, params, sample_points, n_samples=1) -> Array:
    rv = random_variable(gp, params, sample_points)
    return rv.sample(sample_shape=(n_samples,), seed=key)


@dispatch(jnp.DeviceArray, tfd.Distribution)
def sample(key: jnp.DeviceArray, random_variable: tfd.Distribution, n_samples: int = 1) -> Array:
    return random_variable.sample(sample_shape=(n_samples,), seed=key)
