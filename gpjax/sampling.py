from .types import Array
from .gps import Prior, ConjugatePosterior
from .kernel import gram
from .predict import mean, variance
from .utils import I
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd
import jax.numpy as jnp


@dispatch(Prior, dict, jnp.DeviceArray)
def random_variable(gp: Prior,
                    params: dict,
                    sample_points: Array,
                    jitter_amount: float = 1e-6) -> tfd.Distribution:
    mu = gp.mean_function(sample_points)
    gram_matrix = params['variance']*gram(gp.kernel, sample_points/params['lengthscale'])
    jitter_matrix = I(sample_points.shape[0]) * jitter_amount
    covariance = gram_matrix + jitter_matrix
    return tfd.MultivariateNormalFullCovariance(mu.squeeze(), covariance)


@dispatch(ConjugatePosterior, dict, jnp.DeviceArray, jnp.DeviceArray,
          jnp.DeviceArray)
def random_variable(gp: ConjugatePosterior,
                    params: dict,
                    sample_points: Array,
                    train_inputs: Array,
                    train_outputs: Array,
                    jitter_amount: float = 1e-6) -> tfd.Distribution:
    n = sample_points.shape[0]
    # TODO: Return kernel matrices here to avoid replicated computation.
    mu = mean(gp, params, sample_points, train_inputs, train_outputs)
    cov = variance(gp, params, sample_points, train_inputs, train_outputs)
    return tfd.MultivariateNormalFullCovariance(mu.squeeze(),
                                                cov + I(n) * jitter_amount)


@dispatch(jnp.DeviceArray, Prior, dict, jnp.DeviceArray)
def sample(key, gp, params, sample_points, n_samples=1) -> Array:
    rv = random_variable(gp, params, sample_points)
    return rv.sample(sample_shape=(n_samples, ), seed=key)


@dispatch(jnp.DeviceArray, tfd.Distribution)
def sample(key: jnp.DeviceArray,
           random_variable: tfd.Distribution,
           n_samples: int = 1) -> Array:
    return random_variable.sample(sample_shape=(n_samples, ), seed=key)
