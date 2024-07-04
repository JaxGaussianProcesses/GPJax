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
from gpjax.likelihoods import AbstractLikelihood
from gpjax.integrators import AbstractIntegrator
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
    




@dataclass
class Exponential(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> tfd.Distribution:
        r"""The link function of the Poisson likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            tfd.Distribution: The likelihood function.
        """
        assert f.shape[0] == 1
        rate = jnp.clip(f[0,:], a_min=1e-6)
        # rate = jnp.exp(-f[0,:])
        return tfd.Exponential(rate=rate)


    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        raise NotImplementedError



@dataclass
class Gamma(AbstractLikelihood):
    
    scale1: Union[ScalarFloat, Float[Array, "N"]] = param_field(
        jnp.array(1.0, dtype=jnp.float64), bijector=tfb.Softplus()
    )
    
    def link_function(self, f: Float[Array, "L n"]) -> tfd.Distribution:
        assert jnp.shape(f)[0]==1
        rate = jnp.clip(f[0,:], a_min=1e-6)
        # rate = jnp.exp(-f[0,:])
        return tfd.Gamma(concentration=self.scale1, rate=rate)


    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        raise NotImplementedError





@dataclass
class Gamma2(AbstractLikelihood):
    integrator: AbstractIntegrator = static_field(gpx.integrators.TwoDimGHQuadratureIntegrator())
    initial_scale: Union[ScalarFloat, Float[Array, "N"]] = static_field(jnp.array(1.0))

    def link_function(self, f: Float[Array, "L n"]) -> tfd.Distribution:
        assert jnp.shape(f)[0]==2
        rate = jnp.clip(f[0,:], a_min=1e-6)
        concentration = jnp.clip(f[1,:], a_min=1e-6)
        #return tfd.Gamma(concentration=self.initial_scale*jnp.exp(f[1,:]), rate=jnp.exp(-f[0,:]))
        return tfd.Gamma(concentration=concentration, rate=rate)

    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        raise NotImplementedError

from jax import custom_jvp

class FiddleMixture(tfd.Mixture):

    def log_prob(self, x): # [B, 1]
        log_probs = jnp.log(self.cat.probs)[...,1] # [n]
        log_gamma_probs = self.components[0].log_prob(jnp.clip(x,1e-20)) # zeros will be masked anyqay
    
        return _log_prob(x, log_probs, log_gamma_probs)
        
    
@custom_jvp
def _log_prob(x, a, b):
    return  jnp.where(x==0,a, (1-a)*b)
    
@_log_prob.defjvp
def _log_prop_jvp(primals, tangents):
    x, a, b = primals
    x_dot, a_dot, b_dot = tangents
    primal_out = _log_prob(x, a, b)
    tangent_out =  jnp.where(x==0,a_dot, b_dot*(1-a) -a_dot*b) +x_dot # NOTE THIS DOESNT WORK FOR Xdot
    return primal_out, tangent_out
    
@dataclass
class BernoulliGamma(AbstractLikelihood):
    integrator: AbstractIntegrator = static_field(gpx.integrators.ThreeDimGHQuadratureIntegrator())
    def link_function(self, f: Float[Array, "L n"]) -> tfd.Distribution:
        assert jnp.shape(f)[0]==3
        prob = jnp.clip(f[0,:], a_min=1e-6, a_max=1.0-1e-6)
        rate = jnp.clip(f[1,:], a_min=1e-6)
        concentration = jnp.clip(f[2,:], a_min=1e-6)
        
        gamma = tfd.Gamma(concentration=concentration, rate=rate)
        bernoulli_gamma = FiddleMixture(
            cat=tfd.Categorical(probs=jnp.stack([prob, 1.-prob],-1)),
                components=[gamma,tfd.Deterministic(jnp.zeros_like(gamma.mean()))]
                )
        return bernoulli_gamma
    
    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        raise NotImplementedError
    
    
@dataclass
class BernoulliExponential(AbstractLikelihood):
    integrator: AbstractIntegrator = static_field(gpx.integrators.TwoDimGHQuadratureIntegrator())
    def link_function(self, f: Float[Array, "L n"]) -> tfd.Distribution:
        assert jnp.shape(f)[0]==2
        prob = jnp.clip(f[0,:], a_min=1e-6, a_max=1.0-1e-6)
        rate = jnp.clip(f[1,:], a_min=1e-6)
        
        gamma = tfd.Exponential(rate=rate)
        bernoulli_gamma = FiddleMixture(
            cat=tfd.Categorical(probs=jnp.stack([prob, 1.-prob],-1)),
                components=[gamma,tfd.Deterministic(jnp.zeros_like(gamma.mean()))]
                )
        return bernoulli_gamma
    
    def predict(self, dist: tfd.Distribution) -> tfd.Distribution:
        raise NotImplementedError