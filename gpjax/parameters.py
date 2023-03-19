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

from copy import deepcopy
from typing import Dict

import jax


################################
# Base operations
################################
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
# Priors
################################
# TODO: Can this be moved into JaxUtils
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


# def prior_checks(priors: Dict) -> Dict:
#     """
#     Run checks on the parameters' prior distributions. This checks that for
#     Gaussian processes that are constructed with non-conjugate likelihoods, the
#     prior distribution on the function's latent values is a unit Gaussian.

#     Args:
#         priors (Dict): Dictionary specifying the parameters' prior distributions.

#     Returns:
#         Dict: Dictionary specifying the parameters' prior distributions.
#     """
#     if "latent" in priors.keys():
#         latent_prior = priors["latent"]
#         if latent_prior is not None:
#             if not isinstance(latent_prior, dx.Normal):
#                 warnings.warn(
#                     f"A {type(latent_prior)} distribution prior has been placed"
#                     " on the latent function. It is strongly advised that a"
#                     " unit Gaussian prior is used."
#                 )
#         else:
#             warnings.warn("Placing unit Gaussian prior on latent function.")
#             priors["latent"] = dx.Normal(loc=0.0, scale=1.0)
#     else:
#         priors["latent"] = dx.Normal(loc=0.0, scale=1.0)

#     return priors


__all__ = [
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
