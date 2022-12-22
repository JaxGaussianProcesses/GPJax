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

import jax
import distrax as dx
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
from ml_collections import ConfigDict

__config = None

Identity = dx.Lambda(forward=lambda x: x, inverse=lambda x: x)
Softplus = dx.Lambda(
    forward=lambda x: jnp.log(1 + jnp.exp(x)),
    inverse=lambda x: jnp.log(jnp.exp(x) - 1.0),
)


def reset_global_config() -> None:
    global __config
    __config = get_default_config()


def get_global_config() -> ConfigDict:
    """Get the global config file used within GPJax.

    Returns:
        ConfigDict: A `ConfigDict` describing parameter transforms and default values.
    """
    global __config

    if __config is None:
        __config = get_default_config()
        return __config

    # If the global config is available, check if the x64 state has changed
    x64_state = jax.config.x64_enabled

    # If the x64 state has not changed, return the existing global config
    if x64_state is __config.x64_state:
        return __config

    # If the x64 state has changed, return the updated global config
    update_x64_sensitive_settings()
    return __config


def update_x64_sensitive_settings() -> None:
    """Update the global config if x64 state changes."""
    global __config

    # Update the x64 state
    x64_state = jax.config.x64_enabled
    __config.x64_state = x64_state

    # Update the x64 sensitive bijectors
    FillScaleTriL = dx.Chain(
        [
            tfb.FillScaleTriL(diag_shift=jnp.array(__config.jitter)),
        ]
    )

    transformations = __config.transformations
    transformations.triangular_transform = FillScaleTriL


def get_default_config() -> ConfigDict:
    """Construct and return the default config file.

    Returns:
        ConfigDict: A `ConfigDict` describing parameter transforms and default values.
    """

    config = ConfigDict(type_safe=False)
    config.key = jr.PRNGKey(123)

    # Set the x64 state
    config.x64_state = jax.config.x64_enabled

    # Covariance matrix stabilising jitter
    config.jitter = 1e-6

    FillScaleTriL = dx.Chain(
        [
            tfb.FillScaleTriL(diag_shift=jnp.array(config.jitter)),
        ]
    )

    # Default bijections
    config.transformations = transformations = ConfigDict()
    transformations.positive_transform = Softplus
    transformations.identity_transform = Identity
    transformations.triangular_transform = FillScaleTriL

    # Default parameter transforms
    transformations.alpha = "positive_transform"
    transformations.lengthscale = "positive_transform"
    transformations.variance = "positive_transform"
    transformations.smoothness = "positive_transform"
    transformations.shift = "positive_transform"
    transformations.obs_noise = "positive_transform"
    transformations.latent = "identity_transform"
    transformations.basis_fns = "identity_transform"
    transformations.offset = "identity_transform"
    transformations.inducing_inputs = "identity_transform"
    transformations.variational_mean = "identity_transform"
    transformations.variational_root_covariance = "triangular_transform"
    transformations.natural_vector = "identity_transform"
    transformations.natural_matrix = "identity_transform"
    transformations.expectation_vector = "identity_transform"
    transformations.expectation_matrix = "identity_transform"

    return config


# This function is created for testing purposes only
def get_global_config_if_exists() -> ConfigDict:
    """Get the global config file used within GPJax if it is available.

    Returns:
        ConfigDict: A `ConfigDict` describing parameter transforms and default values.
    """
    global __config
    return __config


def add_parameter(param_name: str, bijection: dx.Bijector) -> None:
    """Add a parameter and its corresponding transform to GPJax's config file.

    Args:
        param_name (str): The name of the parameter that is to be added.
        bijection (dx.Bijector): The bijection that should be used to unconstrain the parameter's value.
    """
    lookup_name = f"{param_name}_transform"
    get_global_config()
    __config.transformations[lookup_name] = bijection
    __config.transformations[param_name] = lookup_name
