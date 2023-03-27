import dataclasses

import pytest

from mytree import Identity, Softplus, param_field


@pytest.mark.parametrize("bijector", [Identity, Softplus])
@pytest.mark.parametrize("trainable", [True, False])
def test_param(bijector, trainable):
    param_field_ = param_field(bijector=bijector, trainable=trainable)
    assert isinstance(param_field_, dataclasses.Field)
    assert param_field_.metadata["bijector"] == bijector
    assert param_field_.metadata["trainable"] == trainable

    with pytest.raises(ValueError):
        param_field(
            bijector=bijector, trainable=trainable, metadata={"trainable": trainable}
        )

    with pytest.raises(ValueError):
        param_field(
            bijector=bijector, trainable=trainable, metadata={"bijector": bijector}
        )

    with pytest.raises(ValueError):
        param_field(
            bijector=bijector,
            trainable=trainable,
            metadata={"bijector": Softplus, "trainable": trainable},
        )

    with pytest.raises(ValueError):
        param_field(
            bijector=bijector, trainable=trainable, metadata={"pytree_node": True}
        )

    with pytest.raises(ValueError):
        param_field(
            bijector=bijector, trainable=trainable, metadata={"pytree_node": False}
        )
