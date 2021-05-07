from typing import Tuple

import tensorflow_probability.substrates.jax.bijectors as tfb
from ml_collections import ConfigDict


def get_defaults():
    config = ConfigDict()
    # Covariance matrix stabilising jitter
    config.jitter = 1e-6

    # Default bijections
    config.transformations = transformations = ConfigDict()
    transformations.positive_transform = tfb.Softplus()
    transformations.identity_transform = tfb.Identity()

    # Default parameter transforms
    transformations.lengthscale = "positive_transform"
    transformations.variance = "positive_transform"
    transformations.obs_noise = "positive_transform"
    transformations.latent = "identity_transform"
    transformations.basis_fns = "identity_transform"
    return config


def add_parameter(config: ConfigDict, bijection_tuple: Tuple[str, tfb.Bijector]) -> ConfigDict:
    param_name, bijection = bijection_tuple
    lookup_name = f"custom_{param_name}"
    config.transformations[lookup_name] = bijection
    config.transformations[param_name] = lookup_name
    return config
