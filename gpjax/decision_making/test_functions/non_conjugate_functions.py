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
from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    Integer,
)

from gpjax.dataset import Dataset
from gpjax.decision_making.search_space import ContinuousSearchSpace
from gpjax.typing import KeyArray


@dataclass
class PoissonTestFunction:
    """
    Test function for GPs utilising the Poisson likelihood. Function taken from
    https://docs.jaxgaussianprocesses.com/examples/poisson/#dataset.

    Attributes:
        search_space (ContinuousSearchSpace): Search space for the function.
    """

    search_space = ContinuousSearchSpace(
        lower_bounds=jnp.array([-2.0]),
        upper_bounds=jnp.array([2.0]),
    )

    def generate_dataset(self, num_points: int, key: KeyArray) -> Dataset:
        """
        Generate a toy dataset from the test function.

        Args:
            num_points (int): Number of points to sample.
            key (KeyArray): JAX PRNG key.

        Returns:
            Dataset: Dataset of points sampled from the test function.
        """
        X = self.search_space.sample(num_points=num_points, key=key)
        y = self.evaluate(X)
        return Dataset(X=X, y=y)

    def generate_test_points(
        self, num_points: int, key: KeyArray
    ) -> Float[Array, "N D"]:
        """
        Generate test points from the search space of the test function.

        Args:
            num_points (int): Number of points to sample.
            key (KeyArray): JAX PRNG key.

        Returns:
            Float[Array, 'N D']: Test points sampled from the search space.
        """
        return self.search_space.sample(num_points=num_points, key=key)

    @abstractmethod
    def evaluate(self, x: Float[Array, "N 1"]) -> Integer[Array, "N 1"]:
        """
        Evaluate the test function at a set of points. Function taken from
        https://docs.jaxgaussianprocesses.com/examples/poisson/#dataset.

        Args:
            x (Float[Array, 'N D']): Points to evaluate the test function at.

        Returns:
            Integer[Array, 'N 1']: Values of the test function at the points.
        """
        key = jr.PRNGKey(42)
        f = lambda x: 2.0 * jnp.sin(3 * x) + 0.5 * x
        return jr.poisson(key, jnp.exp(f(x)))
