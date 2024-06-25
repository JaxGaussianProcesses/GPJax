import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.flax_base.types import BijectorLookupType

Bijectors: BijectorLookupType = {
    "real": tfb.Identity(),
    "positive": tfb.Softplus(),
}
