import abc
from typing import Callable, Dict, Optional

import distrax as dx
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from chex import dataclass

from .config import Identity, Softplus, add_parameter
from .gps import Posterior
from .types import Array, Dataset
from .utils import I, concat_dictionaries

Diagonal = dx.Lambda(forward=lambda x: jnp.diagflat(x), inverse=lambda x: jnp.diagonal(x))

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
    mu: Optional[Array] = None
    sqrt: Optional[Array] = None
    diag: Optional[bool] = False
    whiten: Optional[bool] = True

    def __post_init__(self):
        """Initialize the variational family."""
        self.num_inducing = self.inducing_inputs.shape[0]
        add_parameter("inducing_inputs", Identity)

        nz = self.num_inducing

        if self.mu is None:
            self.mu = jnp.zeros((nz, 1))
            add_parameter("mu", Identity)

        if self.sqrt is None:
            self.sqrt = I(nz)
            if self.diag:
                add_parameter("sqrt", FillDiagonal)
            else:
                add_parameter("sqrt", FillTriangular)

    @property
    def params(self) -> Dict:
        hyperparams = {"inducing_inputs": self.inducing_inputs, "mu": self.mu, "sqrt": self.sqrt}
        return hyperparams


@dataclass
class VariationalPosterior(Posterior):
    @abc.abstractmethod
    def mean(self, train_data: Dataset, params: dict) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def variance(self, train_data: Dataset, params: dict) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @property
    def params(self) -> dict:
        return concat_dictionaries(self.prior.params, {"likelihood": self.likelihood.params})

    @abc.abstractmethod
    def elbo(self, train_data: Dataset, transformations: Dict) -> Callable[[Array], Array]:
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
