import abc
from typing import Optional, Dict, Callable
import jax.numpy as jnp
from chex import dataclass

import tensorflow_probability.substrates.jax.bijectors as tfb

from .gps import Posterior
from .types import Array, Dataset
from .config import add_parameter, Identity, Softplus
from .utils import I, concat_dictionaries

import distrax

Diagonal = distrax.Lambda(forward = lambda x: jnp.diagflat(x),
                          inverse = lambda x: jnp.diagonal(x))

FillDiagonal =  distrax.Chain([Diagonal, Softplus])
FillTriangular = distrax.Chain([tfb.FillTriangular()])

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
    
    # Initiliase Gaussian posterior objects:
    def __post_init__(self):
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
        hyperparams = {"inducing_inputs": self.inducing_inputs,
                       "mu": self.mu, "sqrt": self.sqrt}
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

from jax import jit
def ELBO(model: VariationalPosterior, train_data: Dataset, transformations: Dict) -> Callable[[Array], Array]:
    @jit
    def elbo(*args, **kwargs):
        return model.elbo(train_data, transformations)(*args, **kwargs)
    return elbo

def VFE(model: VariationalPosterior, train_data: Dataset, transformations: Dict) -> Callable[[Array], Array]:
    @jit
    def vfe(*args, **kwargs):
        return -model.elbo(train_data, transformations)(*args, **kwargs)
    return vfe