# Copyright 2023 The GPJax Contributors. All Rights Reserved.
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
from beartype.roar import BeartypeCallHintParamViolation
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
)
import pytest

from gpjax.decision_making.search_space import (
    AbstractSearchSpace,
    ContinuousSearchSpace,
)

config.update("jax_enable_x64", True)


def test_abstract_search_space():
    with pytest.raises(TypeError):
        AbstractSearchSpace()


def test_continuous_search_space_empty_bounds():
    with pytest.raises(ValueError):
        ContinuousSearchSpace(lower_bounds=jnp.array([]), upper_bounds=jnp.array([]))


@pytest.mark.parametrize(
    "lower_bounds, upper_bounds",
    [
        (jnp.array([0.0], dtype=jnp.float64), jnp.array([1.0], jnp.float32)),
        (jnp.array([0.0], dtype=jnp.float32), jnp.array([1.0], jnp.float64)),
    ],
)
def test_continuous_search_space_dtype_consistency(
    lower_bounds: Float[Array, " D"], upper_bounds: Float[Array, " D"]
):
    with pytest.raises(ValueError):
        ContinuousSearchSpace(lower_bounds=lower_bounds, upper_bounds=upper_bounds)


@pytest.mark.parametrize(
    "lower_bounds, upper_bounds",
    [
        (jnp.array([0.0]), jnp.array([1.0, 1.0])),
        (jnp.array([0.0, 0.0]), jnp.array([1.0])),
    ],
)
def test_continous_search_space_bounds_shape_consistency(
    lower_bounds: Float[Array, " D1"], upper_bounds: Float[Array, " D2"]
):
    with pytest.raises((BeartypeCallHintParamViolation, ValueError)):
        ContinuousSearchSpace(lower_bounds=lower_bounds, upper_bounds=upper_bounds)


@pytest.mark.parametrize(
    "lower_bounds, upper_bounds",
    [
        (jnp.array([1.0]), jnp.array([0.0])),
        (jnp.array([1.0, 1.0]), jnp.array([0.0, 2.0])),
        (jnp.array([1.0, 1.0]), jnp.array([2.0, 0.0])),
    ],
)
def test_continuous_search_space_bounds_values_consistency(
    lower_bounds: Float[Array, " D"], upper_bounds: Float[Array, " D"]
):
    with pytest.raises(ValueError):
        ContinuousSearchSpace(lower_bounds=lower_bounds, upper_bounds=upper_bounds)


@pytest.mark.parametrize(
    "continuous_search_space, dimensionality",
    [
        (ContinuousSearchSpace(jnp.array([0.0]), jnp.array([1.0])), 1),
        (ContinuousSearchSpace(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])), 2),
        (
            ContinuousSearchSpace(
                jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 1.0, 1.0])
            ),
            3,
        ),
    ],
)
def test_continuous_search_space_dimensionality(
    continuous_search_space: ContinuousSearchSpace, dimensionality: int
):
    assert continuous_search_space.dimensionality == dimensionality


@pytest.mark.parametrize(
    "continuous_search_space",
    [
        ContinuousSearchSpace(jnp.array([0.0]), jnp.array([1.0])),
        ContinuousSearchSpace(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
        ContinuousSearchSpace(jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 1.0, 1.0])),
    ],
)
@pytest.mark.parametrize("num_points", [0, -1])
def test_continous_search_space_invalid_sample_num_points(
    continuous_search_space: ContinuousSearchSpace, num_points: int
):
    with pytest.raises(ValueError):
        continuous_search_space.sample(num_points=num_points, key=jr.key(42))


@pytest.mark.parametrize(
    "continuous_search_space, dimensionality",
    [
        (ContinuousSearchSpace(jnp.array([0.0]), jnp.array([1.0])), 1),
        (ContinuousSearchSpace(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])), 2),
        (
            ContinuousSearchSpace(
                jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 1.0, 1.0])
            ),
            3,
        ),
    ],
)
@pytest.mark.parametrize("num_points", [1, 5, 50])
def test_continuous_search_space_sample_shape(
    continuous_search_space: ContinuousSearchSpace, dimensionality: int, num_points: int
):
    samples = continuous_search_space.sample(num_points=num_points, key=jr.key(42))
    assert samples.shape[0] == num_points
    assert samples.shape[1] == dimensionality


@pytest.mark.parametrize(
    "continuous_search_space",
    [
        ContinuousSearchSpace(jnp.array([0.0]), jnp.array([1.0])),
        ContinuousSearchSpace(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
        ContinuousSearchSpace(jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 1.0, 1.0])),
    ],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(5)])
def test_continous_search_space_sample_same_key_same_samples(
    continuous_search_space: ContinuousSearchSpace, key: jr.key
):
    sample_one = continuous_search_space.sample(num_points=100, key=key)
    sample_two = continuous_search_space.sample(num_points=100, key=key)
    assert jnp.array_equal(sample_one, sample_two)


@pytest.mark.parametrize(
    "continuous_search_space",
    [
        ContinuousSearchSpace(jnp.array([0.0]), jnp.array([1.0])),
        ContinuousSearchSpace(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
        ContinuousSearchSpace(jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 1.0, 1.0])),
    ],
)
@pytest.mark.parametrize(
    "key_one, key_two",
    [(jr.key(42), jr.key(5)), (jr.key(1), jr.key(2))],
)
def test_continuous_search_space_different_keys_different_samples(
    continuous_search_space: ContinuousSearchSpace,
    key_one: jr.key,
    key_two: jr.key,
):
    sample_one = continuous_search_space.sample(num_points=100, key=key_one)
    sample_two = continuous_search_space.sample(num_points=100, key=key_two)
    assert not jnp.array_equal(sample_one, sample_two)


@pytest.mark.parametrize(
    "continuous_search_space",
    [
        ContinuousSearchSpace(
            lower_bounds=jnp.array([0.0]), upper_bounds=jnp.array([1.0])
        ),
        ContinuousSearchSpace(
            lower_bounds=jnp.array([0.0, 0.0]), upper_bounds=jnp.array([1.0, 2.0])
        ),
        ContinuousSearchSpace(
            lower_bounds=jnp.array([0.0, 1.0]), upper_bounds=jnp.array([2.0, 2.0])
        ),
        ContinuousSearchSpace(
            lower_bounds=jnp.array([2.4, 1.7, 4.9]),
            upper_bounds=jnp.array([5.6, 1.8, 6.0]),
        ),
    ],
)
def test_continuous_search_space_valid_sample_ranges(
    continuous_search_space: ContinuousSearchSpace,
):
    samples = continuous_search_space.sample(num_points=100, key=jr.key(42))
    for i in range(continuous_search_space.dimensionality):
        assert jnp.all(samples[:, i] >= continuous_search_space.lower_bounds[i])
        assert jnp.all(samples[:, i] <= continuous_search_space.upper_bounds[i])
