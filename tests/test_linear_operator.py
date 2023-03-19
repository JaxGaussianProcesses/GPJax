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
import jax.tree_util as jtu
import jax.numpy as jnp
from jaxlinop.linear_operator import LinearOperator
from dataclasses import dataclass
from simple_pytree import static_field


def test_covariance_operator() -> None:
    with pytest.raises(TypeError):
        LinearOperator()

@pytest.mark.parametrize("is_dataclass", [True, False])
@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (4, 5, 6)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_instantiate_no_attributes(is_dataclass, shape, dtype) -> None:
    """Test if the abstract operator can be instantiated, given the abstract methods."""

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
        def from_dense(cls, *args, **kwargs):
            pass

    if is_dataclass:
        dataclass(DummyLinearOperator)

    linop = DummyLinearOperator(shape=shape, dtype=dtype)
    assert isinstance(linop, DummyLinearOperator)
    assert isinstance(linop, LinearOperator)
    assert linop.shape == shape
    assert linop.dtype == dtype
    assert jtu.tree_leaves(linop) == [] # shape and dtype are static!


@pytest.mark.parametrize("is_dataclass", [True, False])
@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (4, 5, 6)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_instantiate_with_attributes(is_dataclass, shape, dtype) -> None:
    """Test if the covariance operator can be instantiated with attribute annotations."""

    class DummyLinearOperator(LinearOperator):
        a: int
        b: int = static_field() # Lets have a static attribute here.
        c: int

        def __init__(self, shape, dtype, a=1, b=2, c=3):
            self.a = a
            self.b = b
            self.c = c
            self.shape = shape
            self.dtype = dtype

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
        def from_dense(cls, *args, **kwargs):
            pass

    if is_dataclass:
        dataclass(DummyLinearOperator)

    linop = DummyLinearOperator(shape=shape, dtype=dtype)
    assert isinstance(linop, DummyLinearOperator)
    assert isinstance(linop, LinearOperator)
    assert linop.shape == shape
    assert linop.dtype == dtype
    assert jtu.tree_leaves(linop) == [1, 3] # b, shape, dtype are static!