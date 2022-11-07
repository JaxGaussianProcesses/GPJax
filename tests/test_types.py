# Copyright 2022 The GPJax Contributors. All Rights Reserved.
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

import jax.numpy as jnp
import pytest
from jax.config import config

from gpjax.types import Dataset, NoneType, verify_dataset

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_nonetype():
    assert isinstance(None, NoneType)


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("outd", [1, 2, 10])
@pytest.mark.parametrize("ind", [1, 2, 10])
@pytest.mark.parametrize("n2", [1, 10])
def test_dataset(n, outd, ind, n2):
    x = jnp.ones((n, ind))
    y = jnp.ones((n, outd))
    d = Dataset(X=x, y=y)
    verify_dataset(d)
    assert d.n == n
    assert d.in_dim == ind
    assert d.out_dim == outd

    # test combine datasets
    x2 = 2 * jnp.ones((n2, ind))
    y2 = 2 * jnp.ones((n2, outd))
    d2 = Dataset(X=x2, y=y2)

    d_combined = d + d2
    assert d_combined.n == n + n2
    assert d_combined.in_dim == ind
    assert d_combined.out_dim == outd
    assert (d_combined.y[:n] == 1.0).all()
    assert (d_combined.y[n:] == 2.0).all()
    assert (d_combined.X[:n] == 1.0).all()
    assert (d_combined.X[n:] == 2.0).all()


@pytest.mark.parametrize("nx, ny", [(1, 2), (2, 1), (10, 5), (5, 10)])
def test_dataset_assertions(nx, ny):
    x = jnp.ones((nx, 1))
    y = jnp.ones((ny, 1))
    with pytest.raises(AssertionError):
        ds = Dataset(X=x, y=y)
        verify_dataset(ds)


def test_y_none():
    x = jnp.ones((10, 1))
    d = Dataset(X=x)
    verify_dataset(d)
    assert d.y is None
