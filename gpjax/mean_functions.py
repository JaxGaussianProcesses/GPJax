import abc
from typing import Dict, Optional

import jax.numpy as jnp
from chex import dataclass
from jaxtyping import f64


@dataclass(repr=False)
class AbstractMeanFunction:
    """Abstract mean function that is used to parameterise the Gaussian process."""

    output_dim: Optional[int] = 1
    name: Optional[str] = "Mean function"

    @abc.abstractmethod
    def __call__(self, x: f64["N D"]) -> f64["N Q"]:
        """Evaluate the mean function at the given points. This method is required for all subclasses.

        Args:
            x (Array): The input points at which to evaluate the mean function.

        Returns:
            Array: The mean function evaluated point-wise on the inputs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _initialise_params(self, key: jnp.DeviceArray) -> Dict:
        """Return the parameters of the mean function. This method is required for all subclasses.

        Returns:
            dict: The parameters of the mean function.
        """
        raise NotImplementedError


@dataclass(repr=False)
class Zero(AbstractMeanFunction):
    """
    A zero mean function. This function returns zero for all inputs.
    """

    output_dim: Optional[int] = 1
    name: Optional[str] = "Zero mean function"

    def __call__(self, x: f64["N D"], params: dict) -> f64["N Q"]:
        """Evaluate the mean function at the given points.

        Args:
            x (Array): The input points at which to evaluate the mean function.
            params (dict): The parameters of the mean function.

        Returns:
            Array: A vector of zeros.
        """
        out_shape = (x.shape[0], self.output_dim)
        return jnp.zeros(shape=out_shape)

    def _initialise_params(self, key: jnp.DeviceArray) -> Dict:
        """The parameters of the mean function. For the zero-mean function, this is an empty dictionary."""
        return {}


@dataclass(repr=False)
class Constant(AbstractMeanFunction):
    """
    A zero mean function. This function returns a repeated scalar value for all inputs.
    The scalar value itself can be treated as a model hyperparameter and learned during training.
    """

    output_dim: Optional[int] = 1
    name: Optional[str] = "Constant mean function"

    def __call__(self, x: f64["N D"], params: Dict) -> f64["N Q"]:
        """Evaluate the mean function at the given points.

        Args:
            x (Array): The input points at which to evaluate the mean function.
            params (Dict): The parameters of the mean function.

        Returns:
            Array: A vector of repeated constant values.
        """
        out_shape = (x.shape[0], self.output_dim)
        return jnp.ones(shape=out_shape) * params["constant"]

    def _initialise_params(self, key: jnp.DeviceArray) -> Dict:
        """The parameters of the mean function. For the constant-mean function, this is a dictionary with a single value."""
        return {"constant": jnp.array([1.0])}
