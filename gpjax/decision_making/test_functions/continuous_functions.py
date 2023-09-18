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
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from gpjax.dataset import Dataset
from gpjax.decision_making.search_space import ContinuousSearchSpace
from gpjax.typing import KeyArray


class AbstractContinuousTestFunction(ABC):
    """
    Abstract base class for continuous test functions.

    Attributes:
        search_space (ContinuousSearchSpace): Search space for the function.
        minimizer (Float[Array, '1 D']): Minimizer of the function (to 5 decimal places)
        minimum (Float[Array, '1 1']): Minimum of the function (to 5 decimal places).
    """

    search_space: ContinuousSearchSpace
    minimizer: Float[Array, "1 D"]
    minimum: Float[Array, "1 1"]

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
    def evaluate(self, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        """
        Evaluate the test function at a set of points.

        Args:
            x (Float[Array, 'N D']): Points to evaluate the test function at.

        Returns:
            Float[Array, 'N 1']: Values of the test function at the points.
        """
        raise NotImplementedError


@dataclass
class Forrester(AbstractContinuousTestFunction):
    """
    Forrester function introduced in 'Engineering design via surrogate modelling: a
    practical guide' (Forrester et al. 2008), rescaled to have zero mean and unit
    variance over $`[0, 1]`$.
    """

    search_space = ContinuousSearchSpace(
        lower_bounds=jnp.array([0.0]),
        upper_bounds=jnp.array([1.0]),
    )
    minimizer = jnp.array([[0.75725]])
    minimum = jnp.array([[-1.45280]])

    def evaluate(self, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        mean = 0.45321
        std = jnp.sqrt(19.8577)
        return (((6 * x - 2) ** 2) * jnp.sin(12 * x - 4) - mean) / std


@dataclass
class LogarithmicGoldsteinPrice(AbstractContinuousTestFunction):
    """
    Logarithmic Goldstein-Price function introduced in 'A benchmark of kriging-based
    infill criteria for noisy optimization' (Picheny et al. 2013), which has zero mean
    and unit variance over $`[0, 1]^2`$.
    """

    search_space = ContinuousSearchSpace(
        lower_bounds=jnp.array([0.0, 0.0]),
        upper_bounds=jnp.array([1.0, 1.0]),
    )
    minimizer = jnp.array([[0.5, 0.25]])
    minimum = jnp.array([[-3.12913]])

    def evaluate(self, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        x1 = 4.0 * x[:, 0] - 2.0
        x2 = 4.0 * x[:, 1] - 2.0
        a = 1.0 + (x1 + x2 + 1.0) ** 2 * (
            19.0
            - 14.0 * x1
            + 3.0 * (x1**2)
            - 14.0 * x2
            + 6.0 * x1 * x2
            + 3.0 * (x2**2)
        )
        b = 30.0 + (2.0 * x1 - 3.0 * x2) ** 2 * (
            18.0
            - 32.0 * x1
            + 12.0 * (x1**2)
            + 48.0 * x2
            - 36.0 * x1 * x2
            + 27.0 * (x2**2)
        )
        return ((jnp.log((a * b)) - 8.693) / 2.427).reshape(-1, 1)


@dataclass
class Quadratic(AbstractContinuousTestFunction):
    """
    Toy quadratic function defined over $`[0, 1]`$.
    """

    search_space = ContinuousSearchSpace(
        lower_bounds=jnp.array([0.0]),
        upper_bounds=jnp.array([1.0]),
    )
    minimizer = jnp.array([[0.5]])
    minimum = jnp.array([[0.0]])

    def evaluate(self, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        return (x - 0.5) ** 2
