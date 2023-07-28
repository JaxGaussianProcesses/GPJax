# Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
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

import jax.random as jr
from jaxtyping import Float
import tensorflow_probability.substrates.jax as tfp

from gpjax.typing import (
    Array,
    KeyArray,
)


@dataclass
class AbstractSearchSpace(ABC):
    """The `AbstractSearchSpace` class is an abstract base class for
    search spaces, which are used to bound the domain of functions being optimised.
    """

    @abstractmethod
    def sample_uniform_random(
        self, num_points: int, key: KeyArray
    ) -> Float[Array, "N D"]:
        """Sample points uniformly from the search space.

        Args:
            num_points (int): Number of points to be sampled from the search space.
            key (KeyArray): JAX PRNG key.

        Returns:
            Float[Array, "N D"]: `num_points` points sampled uniformly at random from the search space.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensionality(self) -> int:
        """Dimensionality of the search space.

        Returns:
            int: Dimensionality of the search space.
        """
        raise NotImplementedError


@dataclass
class BoxSearchSpace(AbstractSearchSpace):
    """The `BoxSearchSpace` class is used to bound the domain of functions to
    a box of dimension $`D`$.
    """

    lower_bounds: object  # TODO: Add type annotation for shape (was getting an error from pre-commit)
    upper_bounds: object  # TODO: Add type annotation for shape (was getting an error from pre-commit)

    def __post_init__(self):
        if self.lower_bounds.shape != self.upper_bounds.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")
        if not (self.lower_bounds < self.upper_bounds).all():
            raise ValueError("Lower bounds must be less than upper bounds.")

    @property
    def dimensionality(self) -> int:
        return self.lower_bounds.shape[0]

    def sample_uniform_random(
        self, num_points: int, key: KeyArray
    ) -> Float[Array, "N D"]:
        """Sample points uniformly from the search space.

        Args:
            num_points (int): Number of points to be sampled from the search space.
            key (KeyArray): JAX PRNG key.

        Returns:
            Float[Array, "N D"]: `num_points` points sampled uniformly at random from the search space.
        """
        return jr.uniform(
            key=key,
            shape=(num_points, self.dimensionality),
            minval=self.lower_bounds,
            maxval=self.upper_bounds,
        )

    def sample_halton(self, num_points: int, key: KeyArray) -> Float[Array, "N D"]:
        """Sample points using the Halton sequence from the search space.

        Args:
            num_points (int): Number of points to be sampled from the search space.
            key (KeyArray): JAX PRNG key.

        Returns:
            Float[Array, "N D"]: `num_points` points sampled using the Halton sequence
            from the search space.
        """
        initial_sample = tfp.mcmc.sample_halton_sequence(
            dim=self.dimensionality, num_results=num_points, seed=key
        )  # TODO: Specify dtype=jnp.float64?
        return (
            self.lower_bounds + (self.upper_bounds - self.lower_bounds) * initial_sample
        )
