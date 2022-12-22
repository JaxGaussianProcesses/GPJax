# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import warnings
from copy import deepcopy
from typing import Dict, Tuple
from warnings import warn

import distrax as dx
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import dataclass, PRNGKey as PRNGKeyType
from jaxtyping import Array, Float

from .config import Identity, get_global_config
from .utils import merge_dictionaries


################################
# Base operations
################################
@dataclass
class ParameterState:
    """
    The state of the model. This includes the parameter set, which parameters
    are to be trained and bijectors that allow parameters to be constrained and
    unconstrained.
    """

    params: Dict
    trainables: Dict
    bijectors: Dict

    def unpack(self):
        """Unpack the state into a tuple of parameters, trainables and bijectors.

        Returns:
            Tuple[Dict, Dict, Dict]: The parameters, trainables and bijectors.
        """
        return self.params, self.trainables, self.bijectors


def initialise(model, key: PRNGKeyType = None, **kwargs) -> ParameterState:
    """
    Initialise the stateful parameters of any GPJax object. This function also
    returns the trainability status of each parameter and set of bijectors that
    allow parameters to be constrained and unconstrained.

    Args:
        model: The GPJax object that is to be initialised.
        key (PRNGKeyType, optional): The random key that is to be used for
            initialisation. Defaults to None.

    Returns:
        ParameterState: The state of the model. This includes the parameter
            set, which parameters are to be trained and bijectors that allow
            parameters to be constrained and unconstrained.
    """

    if key is None:
        warn("No PRNGKey specified. Defaulting to seed 123.", UserWarning, stacklevel=2)
        key = jr.PRNGKey(123)
    params = model._initialise_params(key)

    if kwargs:
        _validate_kwargs(kwargs, params)
        for k, v in kwargs.items():
            params[k] = merge_dictionaries(params[k], v)

    bijectors = build_bijectors(params)
    trainables = build_trainables(params)

    return ParameterState(
        params=params,
        trainables=trainables,
        bijectors=bijectors,
    )


def _validate_kwargs(kwargs, params):
    for k, v in kwargs.items():
        if k not in params.keys():
            raise ValueError(f"Parameter {k} is not a valid parameter.")


def recursive_items(d1: Dict, d2: Dict):
    """
    Recursive loop over pair of dictionaries whereby the value of a given key in
    either dictionary can be itself a dictionary.

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


def recursive_complete(d1: Dict, d2: Dict) -> Dict:
    """
    Recursive loop over pair of dictionaries whereby the value of a given key in
    either dictionary can be itself a dictionary. If the value of the key in the
    second dictionary is None, the value of the key in the first dictionary is
    used.

    Args:
        d1 (Dict): The reference dictionary.
        d2 (Dict): The potentially incomplete dictionary.

    Returns:
        Dict: A completed form of the second dictionary.
    """
    for key, value in d1.items():
        if type(value) is dict:
            if key in d2.keys():
                recursive_complete(value, d2[key])
        else:
            if key in d2.keys():
                d1[key] = d2[key]
    return d1


################################
# Parameter transformation
################################
def build_bijectors(params: Dict) -> Dict:
    """
    For each parameter, build the bijection pair that allows the parameter to be
    constrained and unconstrained.

    Args:
        params (Dict): _description_

    Returns:
        Dict: A dictionary that maps each parameter to a bijection.
    """
    bijectors = copy_dict_structure(params)
    config = get_global_config()
    transform_set = config["transformations"]

    def recursive_bijectors_list(ps, bs):
        return [recursive_bijectors(ps[i], bs[i]) for i in range(len(bs))]

    def recursive_bijectors(ps, bs) -> Tuple[Dict, Dict]:
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


def constrain(params: Dict, bijectors: Dict) -> Dict:
    """
    Transform the parameters to the constrained space for corresponding
    bijectors.

    Args:
        params (Dict): The parameters that are to be transformed.
        bijectors (Dict): The bijectors that are to be used for
            transformation.

    Returns:
        Dict: A transformed parameter set. The dictionary is equal in
            structure to the input params dictionary.
    """
    map = lambda param, trans: trans.forward(param)

    return jax.tree_util.tree_map(map, params, bijectors)


def unconstrain(params: Dict, bijectors: Dict) -> Dict:
    """Transform the parameters to the unconstrained space for corresponding
        bijectors.

    Args:
        params (Dict): The parameters that are to be transformed.
        bijectors (Dict): The corresponding dictionary of transforms that
            should be applied to the parameter set.

    Returns:
        Dict: A transformed parameter set. The dictionary is equal in
            structure to the input params dictionary.
    """

    map = lambda param, trans: trans.inverse(param)

    return jax.tree_util.tree_map(map, params, bijectors)


################################
# Priors
################################
def log_density(
    param: Float[Array, "D"], density: dx.Distribution
) -> Float[Array, "1"]:
    """Compute the log density of a parameter given a distribution.

    Args:
        param (Float[Array, "D"]): The parameter that is to be evaluated.
        density (dx.Distribution): The distribution that is to be evaluated.

    Returns:
        Float[Array, "1"]: The log density of the parameter.
    """
    if type(density) == type(None):
        log_prob = jnp.array(0.0)
    else:
        log_prob = jnp.sum(density.log_prob(param))
    return log_prob


def copy_dict_structure(params: Dict) -> Dict:
    """Copy the structure of a dictionary.

    Args:
        params (Dict): The dictionary that is to be copied.

    Returns:
        Dict: A copy of the input dictionary.
    """
    # Copy dictionary structure
    prior_container = deepcopy(params)
    # Set all values to zero
    prior_container = jax.tree_util.tree_map(lambda _: None, prior_container)
    return prior_container


def structure_priors(params: Dict, priors: Dict) -> Dict:
    """First create a dictionary with equal structure to the parameters.
    Then, for each supplied prior, overwrite the None value if it exists.

    Args:
        params (Dict): [description]
        priors (Dict): [description]

    Returns:
        Dict: [description]
    """
    prior_container = copy_dict_structure(params)
    # Where a prior has been supplied, override the None value by the prior distribution.
    complete_prior = recursive_complete(prior_container, priors)
    return complete_prior


def evaluate_priors(params: Dict, priors: Dict) -> Dict:
    """
    Recursive loop over pair of dictionaries that correspond to a parameter's
    current value and the parameter's respective prior distribution. For
    parameters where a prior distribution is specified, the log-prior density is
    evaluated at the parameter's current value.

    Args: params (Dict): Dictionary containing the current set of parameter
        estimates. priors (Dict): Dictionary specifying the parameters' prior
        distributions.

    Returns:
        Dict: The log-prior density, summed over all parameters.
    """
    lpd = jnp.array(0.0)
    if priors is not None:
        for name, param, prior in recursive_items(params, priors):
            lpd += log_density(param, prior)
    return lpd


def prior_checks(priors: Dict) -> Dict:
    """
    Run checks on the parameters' prior distributions. This checks that for
    Gaussian processes that are constructed with non-conjugate likelihoods, the
    prior distribution on the function's latent values is a unit Gaussian.

    Args:
        priors (Dict): Dictionary specifying the parameters' prior distributions.

    Returns:
        Dict: Dictionary specifying the parameters' prior distributions.
    """
    if "latent" in priors.keys():
        latent_prior = priors["latent"]
        if latent_prior is not None:
            if not isinstance(latent_prior, dx.Normal):
                warnings.warn(
                    f"A {type(latent_prior)} distribution prior has been placed on"
                    " the latent function. It is strongly advised that a"
                    " unit Gaussian prior is used."
                )
        else:
            warnings.warn("Placing unit Gaussian prior on latent function.")
            priors["latent"] = dx.Normal(loc=0.0, scale=1.0)
    else:
        priors["latent"] = dx.Normal(loc=0.0, scale=1.0)

    return priors


def build_trainables(params: Dict, status: bool = True) -> Dict:
    """
    Construct a dictionary of trainable statuses for each parameter. By default,
    every parameter within the model is trainable.

    Args:
        params (Dict): The parameter set for which trainable statuses should be
            derived from.
        status (bool): The status of each parameter. Default is True.

    Returns:
        Dict: A dictionary of boolean trainability statuses. The dictionary is
            equal in structure to the input params dictionary.
    """
    # Copy dictionary structure
    prior_container = deepcopy(params)
    # Set all values to zero
    prior_container = jax.tree_util.tree_map(lambda _: status, prior_container)
    return prior_container


def _stop_grad(param: Dict, trainable: Dict) -> Dict:
    """
    When taking a gradient, we want to stop the gradient from flowing through a
    parameter if it is not trainable. This is achieved using the model's
    dictionary of parameters and the corresponding trainability status.

    Args:
        param (Dict): The parameter set for which trainable statuses should be
            derived from.
        trainable (Dict): A boolean value denoting the training status the `param`.

    Returns:
        Dict: The gradient is stopped for non-trainable parameters.
    """
    return jax.lax.cond(trainable, lambda x: x, jax.lax.stop_gradient, param)


def trainable_params(params: Dict, trainables: Dict) -> Dict:
    """
    Stop the gradients flowing through parameters whose trainable status is
    False.

    Args:
        params (Dict): The parameter set for which trainable statuses should
            be derived from.
        trainables (Dict): A dictionary of boolean trainability statuses. The
            dictionary is equal in structure to the input params dictionary.

    Returns:
        Dict: A dictionary parameters. The dictionary is equal in structure to
            the input params dictionary.
    """
    return jax.tree_util.tree_map(
        lambda param, trainable: _stop_grad(param, trainable), params, trainables
    )


__all__ = [
    "ParameterState",
    "initialise",
    "recursive_items",
    "recursive_complete",
    "build_bijectors",
    "constrain",
    "unconstrain",
    "log_density",
    "copy_dict_structure",
    "structure_priors",
    "evaluate_priors",
    "prior_checks",
    "build_trainables",
    "trainable_params",
]
