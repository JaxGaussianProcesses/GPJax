import distrax as dx
import jax.numpy as jnp
import jax.random as jr
from ml_collections import ConfigDict

__config = None

Identity = dx.Lambda(lambda x: x)
Softplus = dx.Lambda(lambda x: jnp.log(1.0 + jnp.exp(x)))


def get_defaults() -> ConfigDict:
    """Construct and globally register the config file used within GPJax.

    Returns:
        ConfigDict: A `ConfigDict` describing parameter transforms and default values.
    """
    config = ConfigDict()
    config.key = jr.PRNGKey(123)
    # Covariance matrix stabilising jitter
    config.jitter = 1e-6

    # Default bijections
    config.transformations = transformations = ConfigDict()
    transformations.positive_transform = Softplus
    transformations.identity_transform = Identity

    # Default parameter transforms
    transformations.lengthscale = "positive_transform"
    transformations.variance = "positive_transform"
    transformations.smoothness = "positive_transform"
    transformations.shift = "positive_transform"
    transformations.obs_noise = "positive_transform"
    transformations.latent = "identity_transform"
    transformations.basis_fns = "identity_transform"
    transformations.offset = "identity_transform"
    global __config
    if not __config:
        __config = config
    return __config


def add_parameter(param_name: str, bijection: dx.Bijector) -> None:
    """Add a parameter and its corresponding transform to GPJax's config file.

    Args:
        param_name (str): The name of the parameter that is to be added.
        bijection (tfb.Bijector): The bijection that should be used to unconstrain the parameter's value.
    """
    lookup_name = f"{param_name}_transform"
    get_defaults()
    __config.transformations[lookup_name] = bijection
    __config.transformations[param_name] = lookup_name
