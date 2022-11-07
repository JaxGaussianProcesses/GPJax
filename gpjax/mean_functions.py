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

import abc
from typing import Dict, Optional

import jax.numpy as jnp
from chex import dataclass
from jaxtyping import Array, Float

from .types import PRNGKeyType


@dataclass(repr=False)
class AbstractMeanFunction:
    """Abstract mean function that is used to parameterise the Gaussian process."""

    output_dim: Optional[int] = 1
    name: Optional[str] = "Mean function"

    @abc.abstractmethod
    def __call__(self, params: Dict, x: Float[Array, "N D"]) -> Float[Array, "N Q"]:
        """Evaluate the mean function at the given points. This method is required for all subclasses.

        Args:
            params (Dict): The parameters of the mean function.
            x (Float[Array, "N D"]): The input points at which to evaluate the mean function.

        Returns:
            Float[Array, "N Q"]: The mean function evaluated point-wise on the inputs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Return the parameters of the mean function. This method is required for all subclasses.

        Args:
            key (PRNGKeyType): The PRNG key to use for initialising the parameters.

        Returns:
            Dict: The parameters of the mean function.
        """
        raise NotImplementedError


@dataclass(repr=False)
class Zero(AbstractMeanFunction):
    """
    A zero mean function. This function returns zero for all inputs.
    """

    output_dim: Optional[int] = 1
    name: Optional[str] = "Zero mean function"

    def __call__(self, params: Dict, x: Float[Array, "N D"]) -> Float[Array, "N Q"]:
        """Evaluate the mean function at the given points.

        Args:
            params (Dict): The parameters of the mean function.
            x (Float[Array, "N D"]): The input points at which to evaluate the mean function.

        Returns:
            Float[Array, "N Q"]: A vector of zeros.
        """
        out_shape = (x.shape[0], self.output_dim)
        return jnp.zeros(shape=out_shape)

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """The parameters of the mean function. For the zero-mean function, this is an empty dictionary.

        Args:
            key (PRNGKeyType): The PRNG key to use for initialising the parameters.

        Returns:
            Dict: The parameters of the mean function.
        """
        return {}


@dataclass(repr=False)
class Constant(AbstractMeanFunction):
    """
    A zero mean function. This function returns a repeated scalar value for all inputs.
    The scalar value itself can be treated as a model hyperparameter and learned during training.
    """

    output_dim: Optional[int] = 1
    name: Optional[str] = "Constant mean function"

    def __call__(self, params: Dict, x: Float[Array, "N D"]) -> Float[Array, "N Q"]:
        """Evaluate the mean function at the given points.

        Args:
            params (Dict): The parameters of the mean function.
            x (Float[Array, "N D"]): The input points at which to evaluate the mean function.

        Returns:
            Float[Array, "N Q"]: A vector of repeated constant values.
        """
        out_shape = (x.shape[0], self.output_dim)
        return jnp.ones(shape=out_shape) * params["constant"]

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """The parameters of the mean function. For the constant-mean function, this is a dictionary with a single value.

        Args:
            key (PRNGKeyType): The PRNG key to use for initialising the parameters.

        Returns:
            Dict: The parameters of the mean function.
        """
        return {"constant": jnp.array([1.0])}


__all__ = [
    "AbstractMeanFunction",
    "Zero",
    "Constant",
]
