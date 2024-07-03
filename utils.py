from dataclasses import dataclass
import tensorflow_probability.substrates.jax.bijectors as tfb
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)
from beartype.typing import Any
from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from gpjax.typing import Array

from abc import ABC
import abc
from jaxtyping import Num, Float
from simple_pytree import Pytree
from gpjax.typing import Array
import gpjax as gpx
from jax import vmap
import tensorflow_probability.substrates.jax.bijectors as tfb
from gpjax.base import Module, param_field,static_field
from gpjax.lower_cholesky import lower_cholesky
import jax.scipy as jsp

import jax
from dataclasses import dataclass
import jax.numpy as jnp
import jax.random as jr
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import tensorflow_probability.substrates.jax.bijectors as tfb

#with install_import_hook("gpjax", "beartype.beartype"):
import gpjax as gpx
from gpjax.distributions import GaussianDistribution

key = jr.PRNGKey(123)

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import tensorflow_probability.substrates.jax.bijectors as tfb



from typing import List, Union, Callable
from jaxtyping import Num, Float
from gpjax.typing import Array, ScalarFloat
from gpjax.dataset import _check_precision
from beartype.typing import Optional
from gpjax.base import Module, param_field,static_field
import cola
from cola.linalg.decompositions.decompositions import Cholesky
from jax import vmap



@dataclass
class ProblemInfo(Pytree):
    num_2d_variables: int = static_field(None)
    num_3d_variables: int = static_field(None)
    num_static_variables: int= static_field(None)
    names_2d_short: List[str]= static_field(None)
    names_3d_short: List[str]= static_field(None)
    names_static_short: List[str]= static_field(None)
    names_2d: List[str]= static_field(None)
    names_3d: List[str]= static_field(None)
    names: List[str]= static_field(None)
    names_short: List[str]= static_field(None)
    names_static: List[str]= static_field(None)
    num_variables: int= static_field(None)
    pressure_levels:  Num[Array, "1 L"]= static_field(None)
    pressure_mean: float= static_field(None)
    pressure_std: float= static_field(None)
    lsm_threshold: float= static_field(None)
    

