# from typing import Iterable, Dict
from collections import Iterable
from chex import Array
import numpyro
import numpyro.distributions as dist
from gpjax.kernels import gram
from gpjax.utils import I
from gpjax.likelihoods import link_function
from multipledispatch import dispatch
from gpjax.gps import ConjugatePosterior, NonConjugatePosterior
from typing import Callable, List
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import module
from copy import deepcopy

numpyro_constraint = numpyro.distributions.constraints.Constraint


@dispatch(dict, numpyro_constraint)
def add_constraints(params: dict, constraint: numpyro_constraint):
    new_params = deepcopy(params)
    for ivariable in new_params.keys():
        new_params[ivariable]["constraint"] = constraint
    return new_params


@dispatch(dict, Iterable, Iterable)
def add_constraints(params: dict, variable: Iterable, constraint: Iterable):
    new_params = deepcopy(params)

    for ivariable, iconstraint in zip(variable, constraint):
        new_params[ivariable]["constraint"] = iconstraint
    return new_params


@dispatch(dict, str, numpyro_constraint)
def add_constraints(params: dict, variable: str, constraint: numpyro_constraint):
    new_params = deepcopy(params)

    new_params[variable]["constraint"] = constraint
    return new_params


def numpyro_dict_params(params: dict) -> List:
    """Function to go from standard dictionary to params"""
    numpyro_params = {}
    for ikey, iparam in params.items():
        numpyro_params[ikey] = {
            "init_value": iparam,
            "constraint": dist.constraints.positive,
        }
    return numpyro_params


@dispatch(ConjugatePosterior, dict)
def numpyro_mle(gp: ConjugatePosterior, numpyro_params: dict) -> Callable:
    def mll(x: Array, y: Array):

        params = {}

        for iname, iparam in numpyro_params.items():

            params[iname] = numpyro.param(
                name=iname,
                init_value=iparam["init_value"],
                constraint=iparam["constraint"],
            )
        # get mean function
        mu = gp.prior.mean_function(x)

        # covariance function
        gram_matrix = gram(gp.prior.kernel, x, params)
        gram_matrix += params["obs_noise"] * I(x.shape[0])

        # scale triangular matrix
        L = jnp.linalg.cholesky(gram_matrix)
        return numpyro.sample(
            "y", dist.MultivariateNormal(loc=mu, scale_tril=L), obs=y.squeeze()
        )

    return mll


def numpyro_dict_priors(params: dict) -> List:
    """Function to go from standard dictionary to params"""
    numpyro_params = {}
    for ikey, iprior in params.items():
        numpyro_params[ikey] = numpyro.sample(name=ikey, fn=iprior)
    return numpyro_params


@dispatch(ConjugatePosterior, dict)
def numpyro_map(gp: ConjugatePosterior, numpyro_params: dict) -> Callable:
    def mll(x: Array, y: Array):

        params = {}
        for ikey, iprior in numpyro_params.items():
            params[ikey] = numpyro.sample(name=ikey, fn=iprior)
        # get mean function
        mu = gp.prior.mean_function(x)

        # covariance function
        gram_matrix = gram(gp.prior.kernel, x, params)
        gram_matrix += params["obs_noise"] * I(x.shape[0])

        # scale triangular matrix
        L = jnp.linalg.cholesky(gram_matrix)
        # return numpyro.sample("y", tfd.MultivariateNormalTriL(mu, L), obs=y.squeeze())
        return numpyro.sample(
            "y", dist.MultivariateNormal(loc=mu, scale_tril=L), obs=y.squeeze()
        )

    return mll


# @dispatch(NonConjugatePosterior, dict, float)
# @dispatch(NonConjugatePosterior, dict)
# def numpyro_marginal_ll(
#     gp: NonConjugatePosterior, init_params: dict, jitter: float = 1e-6
# ) -> Callable:
#     def mll(x: Array, y: Array):

#         # convert parameters
#         params = convert_to_numpyro_params(init_params)

#         # covariance function
#         gram_matrix = gram(gp.prior.kernel, x, params)
#         gram_matrix += I(x.shape[0]) * jitter

#         # scale triangular matrix
#         L = jnp.linalg.cholesky(gram_matrix)
#         F = jnp.matmul(L, params["latent"])

#         # get likelihood function
#         link = link_function(gp.likelihood)
#         rv = link(F)

#         return numpyro.sample("y", rv, obs=y.squeeze())

#     return mll