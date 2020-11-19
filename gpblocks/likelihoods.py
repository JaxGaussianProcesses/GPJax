import jax.numpy as jnp
import jax.scipy as jsp


def gaussian(x, noise):
    lls = jsp.stats.norm.logpdf(x, loc=0.0, scale=noise)
    return jnp.sum(lls)

