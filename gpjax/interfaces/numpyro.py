# from typing import Iterable, Dict
from collections import Iterable
from copy import deepcopy
from typing import Callable, List, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from chex import Array
from jax.scipy.linalg import cholesky
from multipledispatch import dispatch
from numpyro import module

from gpjax.gps import ConjugatePosterior, NonConjugatePosterior
from gpjax.kernels import gram
from gpjax.likelihoods import link_function
from gpjax.utils import I
from gpjax.types import Dataset

numpyro_constraint = numpyro.distributions.constraints.Constraint
numpyro_priors = numpyro.distributions.Distribution


def numpyro_dict_params(params: dict) -> List:
    """Function to go from standard dictionary to params"""
    numpyro_params = {}
    for ikey, iparam in params.items():

        if isinstance(iparam, jnp.DeviceArray) or isinstance(iparam, float):
            numpyro_params[ikey] = {
                "param_type": "param",
                "init_value": iparam,
                "constraint": dist.constraints.positive,
            }

        elif isinstance(iparam, numpyro_priors):
            numpyro_params[ikey] = {
                "param_type": "prior",
                "prior": iparam,
            }

        else:
            raise ValueError(f"Unrecognized param input: {iparam}")

    return numpyro_params


@dispatch(dict, numpyro_constraint)
def add_constraints(params: dict, constraint: numpyro_constraint):
    new_params = deepcopy(params)
    for ivariable in params.keys():
        if new_params[ivariable]["param_type"] == "param":
            new_params[ivariable]["constraint"] = constraint
    return new_params


@dispatch(dict, dict)
def add_constraints(params: dict, constraint: Iterable):
    new_params = deepcopy(params)

    for ivariable, iconstraint in constraint.items():
        if new_params[ivariable]["param_type"] == "param":
            new_params[ivariable]["constraint"] = iconstraint
    return new_params


@dispatch(dict, str, numpyro_constraint)
def add_constraints(params: dict, variable: str, constraint: numpyro_constraint):
    new_params = deepcopy(params)
    if new_params[variable]["param_type"] == "param":
        new_params[variable]["constraint"] = constraint
    return new_params


@dispatch(dict, dict)
def add_priors(params: dict, priors: dict):
    new_params = deepcopy(params)
    for ivariable, iprior in priors.items():
        new_params[ivariable] = {"param_type": "prior", "prior": iprior}
    return new_params


@dispatch(dict, str, numpyro_priors)
def add_priors(params: dict, variable: str, priors: numpyro_priors):
    new_params = deepcopy(params)
    new_params[variable] = {"param_type": "prior", "prior": priors}
    return new_params


@dispatch(dict, numpyro_priors)
def add_priors(params: dict, priors: numpyro_priors):
    new_params = deepcopy(params)
    for ivariable in new_params.keys():
        new_params[ivariable] = {"param_type": "prior", "prior": priors}
    return new_params


@dispatch(ConjugatePosterior, dict)
def numpyro_marginal_ll(gp: ConjugatePosterior, numpyro_params: dict) -> Callable:
    def mll(ds: Dataset):

        x, y = ds.X, ds.y

        params = {}

        for iname, iparam in numpyro_params.items():
            if iparam["param_type"] == "prior":
                params[iname] = numpyro.sample(name=iname, fn=iparam["prior"])
            else:
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
        L = cholesky(gram_matrix, lower=True)
        return numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=mu, scale_tril=L),
            obs=y.squeeze(),
        )

    return mll
