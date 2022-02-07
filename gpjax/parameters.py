import typing as tp
import warnings
from copy import deepcopy

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

from .config import get_defaults
from .types import Array


################################
# Base operations
################################
def initialise(obj) -> tp.Tuple[tp.Dict, tp.Dict, tp.Dict]:
    params = obj.params
    constrainers, unconstrainers = build_transforms(params)
    return params, constrainers, unconstrainers


def recursive_items(d1, d2):
    for key, value in d1.items():
        if type(value) is dict:
            yield from recursive_items(value, d2[key])
        else:
            yield (key, value, d2[key])


def recursive_complete(d1, d2) -> dict:
    for key, value in d1.items():
        if type(value) is dict:
            if key in d2.keys():
                recursive_complete(value, d2[key])
            else:
                pass
        else:
            if key in d2.keys():
                d1[key] = d2[key]
    return d1


# def recursive_fn(d1, d2, fn: tp.Callable[[tp.Any], tp.Any]):
#     for key, value in d1.items():
#         if type(value) is dict:
#             yield from recursive_fn(value, d2[key], fn)
#         else:
#             yield fn(value, d2[key])


################################
# Parameter transformation
################################
def build_transforms(params, key=None) -> tp.Tuple[tp.Dict, tp.Dict]:
    constrainer, unconstrainer = copy_dict_structure(params), copy_dict_structure(params)
    config = get_defaults()
    transform_set = config["transformations"]

    def recursive_transforms(ps, cs, ucs) -> tp.Tuple[tp.Dict, tp.Dict]:
        for key, value in ps.items():
            if type(value) is dict:
                recursive_transforms(value, cs[key], ucs[key])
            else:
                if key in transform_set.keys():
                    transform_type = transform_set[key]
                    bijector = transform_set[transform_type]
                else:
                    bijector = tfp.bijectors.Identity()
                    warnings.warn(
                        f"Parameter {key} has no transform. Defaulting to"
                        " identity transfom."
                    )
                cs[key] = bijector.forward
                ucs[key] = bijector.inverse
        return cs, ucs

    constrainers, unconstrainers = recursive_transforms(
        params, constrainer, unconstrainer
    )
    return constrainers, unconstrainers


def transform(params: dict, transform_map: dict):
    transformed_params = copy_dict_structure(params)

    def apply_transform(untransformed_params, transformed_params, transform_map):
        for key, value in untransformed_params.items():
            if type(value) is dict:
                apply_transform(value, transformed_params[key], transform_map[key])
            else:
                transformed_params[key] = transform_map[key](value)
        return transformed_params

    return apply_transform(params, transformed_params, transform_map)


################################
# Priors
################################
def log_density(param: jnp.DeviceArray, density: tfd.Distribution) -> Array:
    if type(density) == type(None):
        log_prob = jnp.array(0.0)
    else:
        log_prob = jnp.sum(density.log_prob(param))
    return log_prob


def copy_dict_structure(params: dict) -> dict:
    # Copy dictionary structure
    prior_container = deepcopy(params)
    # Set all values to zero
    prior_container = jax.tree_map(lambda _: None, prior_container)
    return prior_container


def structure_priors(params: dict, priors: dict) -> dict:
    """First create a dictionary with equal structure to the parameters. Then, for each supplied prior, overwrite the None value if it exists.

    Args:
        params (dict): [description]
        priors (dict): [description]

    Returns:
        dict: [description]
    """
    prior_container = copy_dict_structure(params)
    # Where a prior has been supplied, override the None value by the prior distribution.
    complete_prior = recursive_complete(prior_container, priors)
    return complete_prior


def evaluate_priors(params: dict, priors: dict) -> dict:
    """Recursive loop over pair of dictionaries that correspond to a parameter's
    current value and the parameter's respective prior distribution. For
    parameters where a prior distribution is specified, the log-prior density is
    evaluated at the parameter's current value.

    Args: params (dict): Dictionary containing the current set of parameter
        estimates. priors (dict): Dictionary specifying the parameters' prior
        distributions.

    Returns: Array: The log-prior density, summed over all parameters.
    """
    lpd = jnp.array(0.0)
    if priors is not None:
        for name, param, prior in recursive_items(params, priors):
            lpd += log_density(param, prior)
    return lpd


def prior_checks(priors: dict) -> dict:
    if "latent" in priors.keys():
        latent_prior = priors["latent"]
        if isinstance(latent_prior, tfd.Distribution) and latent_prior.name != "Normal":
            warnings.warn(
                f"A {latent_prior.name} distribution prior has been placed on"
                " the latent function. It is strongly advised that a"
                " unit-Gaussian prior is used."
            )
        else:
            if not latent_prior:
                priors["latent"] = tfd.Normal(loc=0.0, scale=1.0)
    else:
        priors["latent"] = tfd.Normal(loc=0.0, scale=1.0)

    return priors
