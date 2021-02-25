from multipledispatch import dispatch
import jax.numpy as jnp
from .types import Array
from .kernel import gram
from .utils import I
from .transforms import Transformation, SoftplusTransformation, transform, untransform
from tensorflow_probability.substrates.jax import distributions as tfd
from .gps import ConjugatePosterior, NonConjugatePosterior
from .prior_densities import log_density
from .likelihoods import link_function
from typing import Callable


@dispatch(ConjugatePosterior)
def marginal_ll(gp: ConjugatePosterior,
                transformation: Transformation = SoftplusTransformation,
                negative: bool = False) -> Callable:
    r"""
    Compute :math:`\log p(y | x, \theta) for a conjugate, or exact, Gaussian process.
    Args:
        x: A set of N X M inputs
        y: A set of N X 1 outputs
    Returns: A multivariate normal distribution
    """
    def mll(params: dict, x: Array, y: Array):
        params = untransform(params, transformation)
        mu = gp.prior.mean_function(x)
        gram_matrix = params['variance'] * gram(gp.prior.kernel,
                                                x / params['lengthscale'])
        gram_matrix += params['obs_noise'] * I(x.shape[0])
        L = jnp.linalg.cholesky(gram_matrix)
        random_variable = tfd.MultivariateNormalTriL(mu, L)
        # TODO: Attach log-prior density sum here
        constant = jnp.array(-1.) if negative else jnp.array(1.)
        return constant * random_variable.log_prob(y.squeeze()).mean()

    return mll


@dispatch(NonConjugatePosterior)
def marginal_ll(gp: NonConjugatePosterior, transformation: Transformation = SoftplusTransformation, negative: bool = False, jitter:float = 1e-6) -> Callable:
    def mll(params: dict, x: Array, y: Array):
        params = untransform(params, transformation)
        n = x.shape[0]
        link = link_function(gp.likelihood)
        gram_matrix = params['variance'] * gram(gp.prior.kernel, x/params['lengthscale'])
        gram_matrix += I(n)*jitter
        L = jnp.linalg.cholesky(gram_matrix)
        F = jnp.matmul(L, params['latent'])
        rv = link(F)
        ll = jnp.sum(rv.log_prob(y))
        # TODO: Attach full log-prior density sum here
        latent_prior = jnp.sum(log_density(params['latent'], tfd.Normal(loc=0., scale=1.)))
        constant = jnp.array(-1.) if negative else jnp.array(1.)
        return constant* (ll + latent_prior)
    return mll
