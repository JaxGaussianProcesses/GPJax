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
    @property
    @abc.abstractmethod
    def params(self) -> Dict:
        raise NotImplementedError


@dataclass
class VariationalGaussian(VariationalFamily):
    inducing_inputs: Array
    name: str = "Gaussian"
    variational_mean: Optional[Array] = None
    variational_root_covariance: Optional[Array] = None
    diag: Optional[bool] = False
    whiten: Optional[bool] = True

    def __post_init__(self):
        """Initialize the variational family."""
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
        hyperparams = {
            "inducing_inputs": self.inducing_inputs,
            "variational_mean": self.variational_mean,
            "variational_root_covariance": self.variational_root_covariance,
        }
        return hyperparams
