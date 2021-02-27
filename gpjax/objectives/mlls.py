from typing import Callable

import jax.numpy as jnp
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd

from ..gps import ConjugatePosterior, NonConjugatePosterior
from ..kernels import gram
from ..likelihoods import link_function
from ..parameters.prior_densities import log_density
from ..parameters.transforms import (SoftplusTransformation, Transformation,
                                     untransform)
from ..types import Array
from ..utils import I


@dispatch(ConjugatePosterior)
def marginal_ll(
    gp: ConjugatePosterior,
    transformation: Transformation = SoftplusTransformation,
    negative: bool = False,
) -> Callable:
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
        gram_matrix = params["variance"] * gram(gp.prior.kernel, x / params["lengthscale"])
        gram_matrix += params["obs_noise"] * I(x.shape[0])
        L = jnp.linalg.cholesky(gram_matrix)
        random_variable = tfd.MultivariateNormalTriL(mu, L)
        # TODO: Attach log-prior density sum here
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)
        return constant * random_variable.log_prob(y.squeeze()).mean()

    return mll


@dispatch(NonConjugatePosterior)
def marginal_ll(
    gp: NonConjugatePosterior,
    transformation: Transformation = SoftplusTransformation,
    negative: bool = False,
    jitter: float = 1e-6,
) -> Callable:
    def mll(params: dict, x: Array, y: Array):
        params = untransform(params, transformation)
        n = x.shape[0]
        link = link_function(gp.likelihood)
        gram_matrix = params["variance"] * gram(gp.prior.kernel, x / params["lengthscale"])
        gram_matrix += I(n) * jitter
        L = jnp.linalg.cholesky(gram_matrix)
        F = jnp.matmul(L, params["latent"])
        rv = link(F)
        ll = jnp.sum(rv.log_prob(y))
        # TODO: Attach full log-prior density sum here
        latent_prior = jnp.sum(log_density(params["latent"], tfd.Normal(loc=0.0, scale=1.0)))
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)
        return constant * (ll + latent_prior)

    return mll
