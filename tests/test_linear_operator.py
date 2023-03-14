# Copyright 2022 The JaxLinOp Contributors. All Rights Reserved.
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

import pytest
import jax.numpy as jnp
from jaxlinop.linear_operator import LinearOperator


def test_covariance_operator() -> None:
    with pytest.raises(TypeError):
        LinearOperator(shape=(1, 1), dtype=jnp.float32)


class DummyLinearOperator(LinearOperator):
    def diagonal(self, *args, **kwargs):
        pass

    def shape(self, *args, **kwargs):
        pass

    def dtype(self, *args, **kwargs):
        pass

    def __mul__(self, *args, **kwargs):
        """Multiply linear operator by scalar."""

    def _add_diagonal(self, *args, **kwargs):
        pass

    def __matmul__(self, *args, **kwargs):
        """Matrix multiplication."""

    def to_dense(self, *args, **kwargs):
        pass

    @classmethod
    def from_dense(self, *args, **kwargs):
        pass


def test_can_instantiate() -> None:
    """Test if the covariance operator can be instantiated."""
    res = DummyLinearOperator(shape=(1, 1), dtype=jnp.float32)

    assert isinstance(res, DummyLinearOperator)
    assert isinstance(res, LinearOperator)
    assert res.shape == (1, 1)
    assert res.dtype == jnp.float32
