import abc
from typing import Dict, Optional

import distrax as dx
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from chex import dataclass

from .config import Identity, Softplus, add_parameter, get_defaults
from .types import Array
from .utils import I

DEFAULT_JITTER = get_defaults()["jitter"]

Diagonal = dx.Lambda(
    forward=lambda x: jnp.diagflat(x), inverse=lambda x: jnp.diagonal(x)
)

FillDiagonal = dx.Chain([Diagonal, Softplus])
FillTriangular = dx.Chain([tfb.FillTriangular()])


@dataclass
class VariationalFamily:
    """Abstract base class used to represent families of distributions that can be used within variational inference."""

    @property
    @abc.abstractmethod
    def params(self) -> Dict:
        """The parameters of the distribution. For example, the multivariate Gaussian would return a mean vector and covariance matrix."""
        raise NotImplementedError


@dataclass
class VariationalGaussian(VariationalFamily):
    """The variational Gaussian family of probability distributions."""

    inducing_inputs: Array
    name: str = "Gaussian"
    variational_mean: Optional[Array] = None
    variational_root_covariance: Optional[Array] = None
    diag: Optional[bool] = False
    whiten: Optional[bool] = True

    def __post_init__(self):
        """Initialize the variational Gaussian distribution."""
        self.num_inducing = self.inducing_inputs.shape[0]
        add_parameter("inducing_inputs", Identity)

        m = self.num_inducing

        if self.variational_mean is None:
            self.variational_mean = jnp.zeros((m, 1))
            add_parameter("variational_mean", Identity)

        if self.variational_root_covariance is None:
            self.variational_root_covariance = I(m)
            if self.diag:
                add_parameter("variational_root_covariance", FillDiagonal)
            else:
                add_parameter("variational_root_covariance", FillTriangular)

    @property
    def params(self) -> Dict:
        """Return the variational mean vector, variational root covariance matrix, and inducing input vecot that parameterise the variational Gaussian distribution."""
        hyperparams = {
            "inducing_inputs": self.inducing_inputs,
            "variational_mean": self.variational_mean,
            "variational_root_covariance": self.variational_root_covariance,
        }
        return hyperparams
