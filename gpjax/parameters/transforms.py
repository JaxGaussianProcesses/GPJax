from collections.abc import KeysView
from typing import Callable, Tuple

import jax.numpy as jnp
from ml_collections import ConfigDict


def build_unconstrain(keys: KeysView, config: ConfigDict) -> Callable:
    """
    Build the transformation map the will transform a set of parameters such that each parameter is now defined on the
    entire real line.

    :param keys: A set of dictionary keys that represent parameters name-key.
    :param config: A configuration dictionary that informs which transformation should be applied to which parameter.
    :return: A callable that will apply the desired transformation(s).
    """
    transforms = {k: config.transformations[config.transformations[k]] for k in keys}

    def unconstrain(params: dict) -> dict:
        return {k: jnp.array(transforms[k].inverse(v)) for k, v in params.items()}

    return unconstrain


def build_constrain(keys: KeysView, config: ConfigDict) -> Callable:
    """
    Build the transformation map the will transform a set of parameters such that each unconstrained parameter
    representation is now defined on the parameter's original, possibly constrained, space.

    :param keys: A set of dictionary keys that represent parameters name-key.
    :param config: A configuration dictionary that informs which transformation should be applied to which parameter.
    :return: A callable that will apply the desired transformation(s).
    """
    transforms = {k: config.transformations[config.transformations[k]] for k in keys}

    def constrain(params: dict) -> dict:
        return {k: jnp.array(transforms[k].forward(v)) for k, v in params.items()}

    return constrain


def build_all_transforms(keys: KeysView, config: ConfigDict) -> Tuple[Callable, Callable]:
    """
    Build both the constraining and unconstraining function mappings.

    :param keys: A set of dictionary keys that represent parameters name-key.
    :param config: A configuration dictionary that informs which transformation should be applied to which parameter.
    :return: A tuple of callables that will apply the desired transformation(s) to both the original and the unconstrained parameter values, in this order.
    """
    unconstrain = build_unconstrain(keys, config)
    constrain = build_constrain(keys, config)
    return constrain, unconstrain
