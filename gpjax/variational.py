import abc
from typing import Callable, Dict, Optional

import distrax as dx
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from chex import dataclass

from gpjax.config import get_defaults

from .config import Identity, Softplus, add_parameter
from .gps import Posterior
from .types import Array, Dataset
from .utils import I, concat_dictionaries

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

        nz = self.num_inducing

        if self.variational_mean is None:
            self.variational_mean = jnp.zeros((nz, 1))
            add_parameter("variational_mean", Identity)

        if self.variational_root_covariance is None:
            self.variational_root_covariance = I(nz)
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


@dataclass
class VariationalPosterior:
    posterior: Posterior
    variational_family: VariationalFamily
    jitter: Optional[float] = DEFAULT_JITTER

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood

    @property
    def params(self) -> Dict:
        hyperparams = concat_dictionaries(
            self.posterior.params,
            {"variational_family": self.variational_family.params},
        )
        return hyperparams

    @abc.abstractmethod
    def mean(self, train_data: Dataset, params: dict) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def variance(self, train_data: Dataset, params: dict) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def elbo(
        self, train_data: Dataset, transformations: Dict
    ) -> Callable[[Array], Array]:
        raise NotImplementedError


def ELBO(
    model: VariationalPosterior, train_data: Dataset, transformations: Dict
) -> Callable[[Array], Array]:
    def elbo(*args, **kwargs):
        return model.elbo(train_data, transformations)(*args, **kwargs)

    return elbo


def VFE(
    model: VariationalPosterior, train_data: Dataset, transformations: Dict
) -> Callable[[Array], Array]:
    def vfe(*args, **kwargs):
        return -model.elbo(train_data, transformations)(*args, **kwargs)

    return vfe
