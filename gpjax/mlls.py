from multipledispatch import dispatch
import jax.numpy as jnp
from .types import Array
from .kernel import gram
from .utils import I
from .transforms import Transformation, SoftplusTransformation, transform, untransform
from tensorflow_probability.substrates.jax import distributions as tfd
from .gps import ExactPosterior
from typing import Callable


@dispatch(ExactPosterior, jnp.DeviceArray, jnp.DeviceArray)
def marginal_ll(gp: ExactPosterior,
                x: Array,
                y: Array,
                transformation: Transformation = SoftplusTransformation,
                negative: bool = False) -> Callable:
    r"""
    Compute :math:`\log p(y | x, \theta) for a conjugate, or exact, Gaussian process.
    Args:
        x: A set of N X M inputs
        y: A set of N X 1 outputs
    Returns: A multivariate normal distribution
    """
    def mll(params: dict):
        params = untransform(params, SoftplusTransformation)
        mu = gp.prior.mean_function(x)
        gram_matrix = params['variance'] * gram(gp.prior.kernel,
                                                x / params['lengthscale'])
        cov = gram_matrix
        cov += params['obs_noise'] * I(x.shape[0])
        L = jnp.linalg.cholesky(cov)
        random_variable = tfd.MultivariateNormalTriL(mu, L)
        # TODO: Attach log-prior density sum here
        constant = jnp.array(-1.) if negative else jnp.array(1.)
        return constant * random_variable.log_prob(y.squeeze()).mean()

    return mll
