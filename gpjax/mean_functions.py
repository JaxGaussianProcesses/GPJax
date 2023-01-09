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
from jax.random import KeyArray
from jaxtyping import Array, Float
from jaxutils import PyTree

import deprecation


class AbstractMeanFunction(PyTree):
    """Abstract mean function that is used to parameterise the Gaussian process."""

    def __init__(
        self, output_dim: Optional[int] = 1, name: Optional[str] = "Mean function"
    ):
        """Initialise the mean function.

        Args:
            output_dim (Optional[int]): The output dimension of the mean function. Defaults to 1.
            name (Optional[str]): The name of the mean function. Defaults to "Mean function".
        """
        self.output_dim = output_dim
        self.name = name

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
    def init_params(self, key: KeyArray) -> Dict:
        """Return the parameters of the mean function. This method is required for all subclasses.

        Args:
            key (KeyArray): The PRNG key to use for initialising the parameters.

        Returns:
            Dict: The parameters of the mean function.
        """
        raise NotImplementedError

    @deprecation.deprecated(
        deprecated_in="0.5.7",
        removed_in="0.6.0",
        details="Use the ``init_params`` method for parameter initialisation.",
    )
    def _initialise_params(self, key: KeyArray) -> Dict:
        """Deprecated method for initialising the GP's parameters. Succeded by ``init_params``."""
        return self.init_params(key)


class Zero(AbstractMeanFunction):
    """
    A zero mean function. This function returns zero for all inputs.
    """

    def __init__(
        self, output_dim: Optional[int] = 1, name: Optional[str] = "Mean function"
    ):
        """Initialise the zero-mean function.

        Args:
            output_dim (Optional[int]): The output dimension of the mean function. Defaults to 1.
            name (Optional[str]): The name of the mean function. Defaults to "Mean function".
        """
        super().__init__(output_dim, name)

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

    def init_params(self, key: KeyArray) -> Dict:
        """The parameters of the mean function. For the zero-mean function, this is an empty dictionary.

        Args:
            key (KeyArray): The PRNG key to use for initialising the parameters.

        Returns:
            Dict: The parameters of the mean function.
        """
        return {}


class Constant(AbstractMeanFunction):
    """
    A zero mean function. This function returns a repeated scalar value for all inputs.
    The scalar value itself can be treated as a model hyperparameter and learned during training.
    """

    def __init__(
        self, output_dim: Optional[int] = 1, name: Optional[str] = "Mean function"
    ):
        """Initialise the constant-mean function.

        Args:
            output_dim (Optional[int]): The output dimension of the mean function. Defaults to 1.
            name (Optional[str]): The name of the mean function. Defaults to "Mean function".
        """
        super().__init__(output_dim, name)

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

    def init_params(self, key: KeyArray) -> Dict:
        """The parameters of the mean function. For the constant-mean function, this is a dictionary with a single value.

        Args:
            key (KeyArray): The PRNG key to use for initialising the parameters.

        Returns:
            Dict: The parameters of the mean function.
        """
        return {"constant": jnp.array([1.0])}


__all__ = [
    "AbstractMeanFunction",
    "Zero",
    "Constant",
]
