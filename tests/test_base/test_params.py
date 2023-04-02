import dataclasses

import pytest

import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.base import param_field


@pytest.mark.parametrize("bijector", [tfb.Identity, tfb.Softplus])
@pytest.mark.parametrize("trainable", [True, False])
def test_param(bijector, trainable):
    param_field_ = param_field(bijector=bijector(), trainable=trainable)
    assert isinstance(param_field_, dataclasses.Field)
    assert isinstance(param_field_.metadata["bijector"], bijector)
    assert param_field_.metadata["trainable"] == trainable

    with pytest.raises(ValueError):
        param_field(
            bijector=bijector(), trainable=trainable, metadata={"trainable": trainable}
        )

    with pytest.raises(ValueError):
        param_field(
            bijector=bijector(), trainable=trainable, metadata={"bijector": bijector()}
        )

    with pytest.raises(ValueError):
        param_field(
            bijector=bijector(),
            trainable=trainable,
            metadata={"bijector": tfb.Softplus(), "trainable": trainable},
        )

    with pytest.raises(ValueError):
        param_field(
            bijector=bijector(), trainable=trainable, metadata={"pytree_node": True}
        )

    with pytest.raises(ValueError):
        param_field(
            bijector=bijector(), trainable=trainable, metadata={"pytree_node": False}
        )
