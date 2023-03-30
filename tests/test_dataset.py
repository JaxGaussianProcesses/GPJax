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

import jax.numpy as jnp
import pytest
from jaxutils.dataset import Dataset


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("outd", [1, 2, 10])
@pytest.mark.parametrize("ind", [1, 2, 10])
@pytest.mark.parametrize("n2", [1, 10])
def test_dataset(n: int, outd: int, ind: int, n2: int) -> None:
    x = jnp.ones((n, ind))
    y = jnp.ones((n, outd))
    d = Dataset(X=x, y=y)

    assert d.n == n
    assert d.in_dim == ind
    assert d.out_dim == outd
    assert (
        d.__repr__()
        == f"- Number of observations: {n}\n- Input dimension: {ind}\n- Output"
        f" dimension: {outd}"
    )

    # Test combine datasets.
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

    # Test supervised and unsupervised.
    assert d.is_supervised() is True
    dunsup = Dataset(y=y)
    assert dunsup.is_unsupervised() is True


@pytest.mark.parametrize("nx, ny", [(1, 2), (2, 1), (10, 5), (5, 10)])
@pytest.mark.parametrize("outd", [1, 2, 10])
@pytest.mark.parametrize("ind", [1, 2, 10])
def test_dataset_assertions(nx: int, ny: int, outd: int, ind: int) -> None:
    x = jnp.ones((nx, ind))
    y = jnp.ones((ny, outd))

    with pytest.raises(ValueError):
        Dataset(X=x, y=y)


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("outd", [1, 2, 10])
@pytest.mark.parametrize("ind", [1, 2, 10])
def test_2d_inputs(n: int, outd: int, ind: int) -> None:
    x = jnp.ones((n, ind))
    y = jnp.ones((n,))

    with pytest.raises(ValueError):
        Dataset(X=x, y=y)

    x = jnp.ones((n,))
    y = jnp.ones((n, outd))

    with pytest.raises(ValueError):
        Dataset(X=x, y=y)


def test_y_none() -> None:
    x = jnp.ones((10, 1))
    d = Dataset(X=x)
    assert d.y is None
