from typing import Tuple

import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
from ml_collections import ConfigDict


__config = None


def get_defaults() -> ConfigDict:
    config = ConfigDict()
    config.key = jr.PRNGKey(123)
    # Covariance matrix stabilising jitter
    config.jitter = 1e-6

    # Default bijections
    config.transformations = transformations = ConfigDict()
    transformations.positive_transform = tfb.Softplus()
    transformations.identity_transform = tfb.Identity()

    # Default parameter transforms
    transformations.lengthscale = "positive_transform"
    transformations.shift = "positive_transform"
    transformations.variance = "positive_transform"
    transformations.obs_noise = "positive_transform"
    transformations.latent = "identity_transform"
    transformations.basis_fns = "identity_transform"
    transformations.offset = "identity_transform"
    global __config
    if not __config:
        __config = config
    return __config


def add_parameter(param_name: str, bijection: tfb.Bijector) -> ConfigDict:
    lookup_name = f"{param_name}_transform"
    if not __config:
        get_defaults()
    __config.transformations[lookup_name] = bijection
    __config.transformations[param_name] = lookup_name
    # return __config
