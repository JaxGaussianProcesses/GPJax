# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
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
