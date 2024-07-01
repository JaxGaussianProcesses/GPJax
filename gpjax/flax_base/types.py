import typing as tp

import tensorflow_probability.substrates.jax.bijectors as tfb

DomainType = tp.Literal["real", "positive"]
A = tp.TypeVar("A")


# class BijectorLookup(tp.TypedDict):
#     domain: DomainType
#     bijector: tfb.Bijector


class BijectorLookupType(tp.Dict[DomainType, tfb.Bijector]):
    pass


__all__ = ["DomainType", "A", "BijectorLookupType"]
