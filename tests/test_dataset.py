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

from dataclasses import is_dataclass

try:
    import beartype

    ValidationErrors = (ValueError, beartype.roar.BeartypeCallHintParamViolation)
except ImportError:
    ValidationErrors = ValueError

from jax.config import config
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from gpjax.dataset import Dataset

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("out_dim", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_dataset_init(n: int, in_dim: int, out_dim: int) -> None:
    # Create dataset
    x = jnp.ones((n, in_dim))
    y = jnp.ones((n, out_dim))

    D = Dataset(X=x, y=y)

    # Test dataset shapes
    assert D.n == n
    assert D.in_dim == in_dim
    assert D.out_dim == out_dim

    # Test representation
    assert (
        D.__repr__()
        == f"- Number of observations: {n}\n- Input dimension: {in_dim}\n- Output"
        f" dimension: {out_dim}"
    )

    # Ensure dataclass
    assert is_dataclass(D)

    # Test supervised and unsupervised
    assert Dataset(X=x, y=y).is_supervised() is True
    assert Dataset(y=y).is_unsupervised() is True

    # Check tree flatten
    assert jtu.tree_leaves(D) == [x, y]


@pytest.mark.parametrize("n1", [1, 2, 10])
@pytest.mark.parametrize("n2", [1, 2, 10])
@pytest.mark.parametrize("out_dim", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_dataset_add(n1: int, n2: int, in_dim: int, out_dim: int) -> None:
    # Create first dataset
    x1 = jnp.ones((n1, in_dim))
    y1 = jnp.ones((n1, out_dim))
    D1 = Dataset(X=x1, y=y1)

    # Create second dataset
    x2 = 2 * jnp.ones((n2, in_dim))
    y2 = 2 * jnp.ones((n2, out_dim))
    D2 = Dataset(X=x2, y=y2)

    # Add datasets
    D = D1 + D2

    # Test shapes
    assert D.n == n1 + n2
    assert D.in_dim == in_dim
    assert D.out_dim == out_dim

    # Test representation
    assert (
        D.__repr__()
        == f"- Number of observations: {n1 + n2}\n- Input dimension: {in_dim}\n- Output"
        f" dimension: {out_dim}"
    )

    # Ensure dataclass
    assert is_dataclass(D)

    # Test supervised and unsupervised
    assert (Dataset(X=x1, y=y1) + Dataset(X=x2, y=y2)).is_supervised() is True
    assert (Dataset(y=y1) + Dataset(y=y2)).is_unsupervised() is True

    # Check tree flatten
    x = jnp.concatenate((x1, x2))
    y = jnp.concatenate((y1, y2))
    (jtu.tree_leaves(D)[0] == x).all()
    (jtu.tree_leaves(D)[1] == y).all()


@pytest.mark.parametrize(("nx", "ny"), [(1, 2), (2, 1), (10, 5), (5, 10)])
@pytest.mark.parametrize("out_dim", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_dataset_incorrect_lengths(nx: int, ny: int, out_dim: int, in_dim: int) -> None:
    # Create input and output pairs of different lengths
    x = jnp.ones((nx, in_dim))
    y = jnp.ones((ny, out_dim))

    # Ensure error is raised upon dataset creation
    with pytest.raises(ValidationErrors):
        Dataset(X=x, y=y)


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("out_dim", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_2d_inputs(n: int, out_dim: int, in_dim: int) -> None:
    # Create dataset where output dimension is incorrectly not 2D
    x = jnp.ones((n, in_dim))
    y = jnp.ones((n,))

    # Ensure error is raised upon dataset creation
    with pytest.raises(ValidationErrors):
        Dataset(X=x, y=y)

    # Create dataset where input dimension is incorrectly not 2D
    x = jnp.ones((n,))
    y = jnp.ones((n, out_dim))

    # Ensure error is raised upon dataset creation
    with pytest.raises(ValidationErrors):
        Dataset(X=x, y=y)


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_y_none(n: int, in_dim: int) -> None:
    # Create a dataset with no output
    x = jnp.ones((n, in_dim))
    D = Dataset(X=x)

    # Ensure is dataclass
    assert is_dataclass(D)

    # Ensure output is None
    assert D.y is None

    # Check tree flatten
    assert jtu.tree_leaves(D) == [x]


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("out_dim", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
def test_dataset_missing(n: int, in_dim: int, out_dim: int) -> None:
    # Create dataset
    x = jnp.ones((n, in_dim))
    y = jr.normal(jr.PRNGKey(123), (n, out_dim))
    y = y.at[y < 0].set(jnp.nan)
    mask = jnp.isnan(y)
    D = Dataset(X=x, y=y)

    # Check mask
    assert D.mask is not None
    assert jnp.array_equal(D.mask, mask)

    # Create second dataset
    x2 = 2 * jnp.ones((n, in_dim))
    y2 = 2 * jnp.ones((n, out_dim))
    D2 = Dataset(X=x2, y=y2)

    # Add datasets
    D2 = D + D2

    # Check mask
    assert jnp.sum(D2.mask) == jnp.sum(D.mask)

    # Test dataset shapes
    assert D.n == n
    assert D.in_dim == in_dim
    assert D.out_dim == out_dim

    # Check tree flatten
    # lexicographic order: uppercase "X" comes before lowercase "m"
    x_, mask_, y_ = jtu.tree_leaves(D)
    assert jnp.allclose(x, x_)
    assert jnp.array_equal(mask, mask_)
    assert jnp.allclose(y, y_, equal_nan=True)


@pytest.mark.parametrize(
    ("prec_x", "prec_y"),
    [
        (jnp.float32, jnp.float64),
        (jnp.float64, jnp.float32),
        (jnp.float32, jnp.float32),
    ],
)
@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("in_dim", [1, 2, 10])
@pytest.mark.parametrize("out_dim", [1, 2, 10])
def test_precision_warning(
    n: int, in_dim: int, out_dim: int, prec_x: jnp.dtype, prec_y: jnp.dtype
) -> None:
    # Create dataset
    x = jnp.ones((n, in_dim)).astype(prec_x)
    y = jnp.ones((n, out_dim)).astype(prec_y)

    # Check for warnings if dtypes are not float64
    expected_warnings = 0
    if prec_x != jnp.float64:
        expected_warnings += 1
    if prec_y != jnp.float64:
        expected_warnings += 1

    with pytest.warns(UserWarning, match=".* is not of type float64.*") as record:
        Dataset(X=x, y=y)

    assert len(record) == expected_warnings
