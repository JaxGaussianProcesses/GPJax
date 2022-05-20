import typing as tp
import warnings
from copy import deepcopy

import distrax as dx
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from .config import get_defaults
from .types import Array

Identity = dx.Lambda(lambda x: x)


################################
# Base operations
################################
def initialise(obj) -> tp.Tuple[tp.Dict, tp.Dict, tp.Dict]:
    """Initialise the stateful parameters of any GPJax object. This function also returns the trainability status of each parameter and set of bijectors that allow parameters to be constrained and unconstrained."""
    params = obj.params
    constrainers, unconstrainers = build_transforms(params)
    trainables = build_trainables(params)
    return params, trainables, constrainers, unconstrainers


def recursive_items(d1: tp.Dict, d2: tp.Dict):
    """Recursive loop over pair of dictionaries whereby the value of a given key in either dictionary can be itself a dictionary.

    Args:
        d1 (_type_): _description_
        d2 (_type_): _description_

    Yields:
        _type_: _description_
    """
    for key, value in d1.items():
        if type(value) is dict:
            yield from recursive_items(value, d2[key])
        else:
            yield (key, value, d2[key])


def recursive_complete(d1: tp.Dict, d2: tp.Dict) -> tp.Dict:
    """Recursive loop over pair of dictionaries whereby the value of a given key in either dictionary can be itself a dictionary. If the value of the key in the second dictionary is None, the value of the key in the first dictionary is used.

    Args:
        d1 (tp.Dict): The reference dictionary.
        d2 (tp.Dict): The potentially incomplete dictionary.

    Returns:
        tp.Dict: A completed form of the second dictionary.
    """
    for key, value in d1.items():
        if type(value) is dict:
            if key in d2.keys():
                recursive_complete(value, d2[key])
            # else:
            #     pass
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
def build_bijectors(params: tp.Dict) -> tp.Dict:
    """For each parameter, build the bijection pair that allows the parameter to be constrained and unconstrained.

    Args:
        params (tp.Dict): _description_

    Returns:
        tp.Dict: A dictionary that maps each parameter to a bijection.
    """
    bijectors = copy_dict_structure(params)
    config = get_defaults()
    transform_set = config["transformations"]

    def recursive_bijectors_list(ps, bs):
        return [recursive_bijectors(ps[i], bs[i]) for i in range(len(bs))]

    def recursive_bijectors(ps, bs) -> tp.Tuple[tp.Dict, tp.Dict]:
        if type(ps) is list:
            bs = recursive_bijectors_list(ps, bs)

        else:
            for key, value in ps.items():
                if type(value) is dict:
                    recursive_bijectors(value, bs[key])
                elif type(value) is list:
                    bs[key] = recursive_bijectors_list(value, bs[key])
                else:
                    if key in transform_set.keys():
                        transform_type = transform_set[key]
                        bijector = transform_set[transform_type]
                    else:
                        bijector = Identity
                        warnings.warn(
                            f"Parameter {key} has no transform. Defaulting to identity transfom."
                        )
                    bs[key] = bijector
        return bs

    return recursive_bijectors(params, bijectors)


def build_transforms(params: tp.Dict) -> tp.Tuple[tp.Dict, tp.Dict]:
    """Using the bijector that is associated with each parameter, construct a pair of functions from the bijector that allow the parameter to be constrained and unconstrained.

    Args:
        params (tp.Dict): The parameter set for which transformations should be derived from.

    Returns:
        tp.Tuple[tp.Dict, tp.Dict]: A pair of dictionaries. The first dictionary maps each parameter to a function that constrains the parameter. The second dictionary maps each parameter to a function that unconstrains the parameter.
    """

    def forward(bijector):
        return bijector.forward

    def inverse(bijector):
        return bijector.inverse

    bijectors = build_bijectors(params)

    constrainers = jax.tree_map(lambda _: forward, deepcopy(params))
    unconstrainers = jax.tree_map(lambda _: inverse, deepcopy(params))

    constrainers = jax.tree_map(lambda f, b: f(b), constrainers, bijectors)
    unconstrainers = jax.tree_map(lambda f, b: f(b), unconstrainers, bijectors)

    return constrainers, unconstrainers


def transform(params: tp.Dict, transform_map: tp.Dict) -> tp.Dict:
    """Transform the parameters according to the constraining or unconstraining function dictionary.

    Args:
        params (tp.Dict): The parameters that are to be transformed.
        transform_map (tp.Dict): The corresponding dictionary of transforms that should be applied to the parameter set.

    Returns:
        tp.Dict: A transformed parameter set.s The dictionary is equal in structure to the input params dictionary.
    """
    return jax.tree_map(lambda param, trans: trans(param), params, transform_map)


################################
# Priors
################################
def log_density(param: jnp.DeviceArray, density: dx.Distribution) -> Array:
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
    """Run checks on th parameters' prior distributions. This checks that for Gaussian processes that are constructed with non-conjugate likelihoods, the prior distribution on the function's latent values is a unit Gaussian."""
    if "latent" in priors.keys():
        latent_prior = priors["latent"]
        if isinstance(latent_prior, dx.Distribution) and latent_prior.name != "Normal":
            warnings.warn(
                f"A {latent_prior.name} distribution prior has been placed on"
                " the latent function. It is strongly advised that a"
                " unit-Gaussian prior is used."
            )
        elif (
            isinstance(latent_prior, tfd.Distribution) and latent_prior.name != "Normal"
        ):
            warnings.warn(
                f"A {latent_prior.name} distribution from Tensorflow Probability has been"
                "placed on the latent function. We advise using a unit-Gaussian prior from"
                " Distrax."
            )
        else:
            if not latent_prior:
                priors["latent"] = dx.Normal(loc=0.0, scale=1.0)
    else:
        priors["latent"] = dx.Normal(loc=0.0, scale=1.0)

    return priors


def build_trainables(params: tp.Dict) -> tp.Dict:
    """Construct a dictionary of trainable statuses for each parameter. By default, every parameter within the model is trainable.

    Args:
        params (tp.Dict): The parameter set for which trainable statuses should be derived from.

    Returns:
        tp.Dict: A dictionary of boolean trainability statuses. The dictionary is equal in structure to the input params dictionary.
    """
    # Copy dictionary structure
    prior_container = deepcopy(params)
    # Set all values to zero
    prior_container = jax.tree_map(lambda _: True, prior_container)
    return prior_container


def stop_grad(param: tp.Dict, trainable: tp.Dict):
    """When taking a gradient, we want to stop the gradient from flowing through a parameter if it is not trainable. This is achieved using the model's dictionary of parameters and the corresponding trainability status."""
    return jax.lax.cond(trainable, lambda x: x, jax.lax.stop_gradient, param)


def trainable_params(params: tp.Dict, trainables: tp.Dict) -> tp.Dict:
    """Stop the gradients flowing through parameters whose trainable status is False"""
    return jax.tree_map(
        lambda param, trainable: stop_grad(param, trainable), params, trainables
    )
