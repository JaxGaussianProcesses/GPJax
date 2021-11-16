import warnings

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from ..gps import NonConjugatePosterior
from ..types import Array


def log_density(param: jnp.DeviceArray, density: tfd.Distribution) -> Array:
    if type(density) == type(None):
        return jnp.array(0.0)
    else:
        return density.log_prob(param)


def evaluate_prior(params: dict, priors: dict) -> Array:
    """Recursive loop over pair of dictionaries that correspond to a parameter's
    current value and the parameter's respective prior distribution. For parameters
    where a prior distribution is specified, the log-prior density is evaluated at the
    parameter's current value.

    Args:
        params (dict): Dictionary containing the current set of parameter estimates.
        priors (dict): Dictionary specifying the parameters' prior distributions.

    Returns:
        Array: The log-prior density, summed over all parameters.
    """
    lpd = jnp.array(0)
    for param, val in priors.items():
        # Deal with the case that the key's value is another dictionary e.g., kernel params
        if isinstance(val, dict):
            for sub_param, sub_val in val.items():
                lpd += log_density(params[param][sub_param], priors[param][sub_param])
        # Deal with the case that the key's value is a list of dictionaries e.g., sum kernels
        elif isinstance(val, list):
            for idx, pair in enumerate(val):
                for sub_param, sub_val in pair.items():
                    lpd += log_density(
                        params[param][idx][sub_param], priors[param][idx][sub_param]
                    )
        # Deal with the case that the key's value is a regular array e.g., latent values.
        else:
            lpd += jnp.sum(log_density(params[param], priors[param]))
    return lpd


def prior_checks(gp: NonConjugatePosterior, priors: dict) -> dict:
    if "latent" in priors.keys():
        latent_prior = priors["latent"]
        if latent_prior.name != "Normal":
            warnings.warn(
                f"A {latent_prior.name} distribution prior has been placed on the latent function. It is strongly advised that a unit-Gaussian prior is used."
            )
        return priors
    else:
        priors["latent"] = tfd.Normal(loc=0.0, scale=1.0)
        return priors
