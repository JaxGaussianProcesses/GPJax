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


from optax import GradientTransformation
import jax.tree_util as jtu


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
    
    
    
    
from models import VariationalPrecipGP
    

def thin_model(problem_info:ProblemInfo, D:gpx.Dataset, model:VariationalPrecipGP, target_num, num_test=100):
    def test_model_without_component(D: gpx.Dataset, model:VariationalPrecipGP, idx:List[int], return_model = False, num_samples=100):
        base_kernels = []
        for j in range(model.num_latents):
            new_base_kernels = []
            for i in range(len(model.base_kernels[j])):
                if i == idx[1] and j == idx[0]:
                    new_base_kernels.append(gpx.kernels.Constant(constant=jnp.array(0.0, dtype=jnp.float64), active_dims=[i]))
                else:
                    new_base_kernels.append(model.base_kernels[j][i])
                    
            base_kernels.append(new_base_kernels)
        new_model = model.replace(base_kernels=base_kernels)
        if return_model:
            return new_model 
        mean, var = new_model.predict_indiv(D.X[-num_test:,:])
        samples_f = jnp.stack([tfd.MultivariateNormalDiag(mean.T[i:i+1], jnp.sqrt(var.T[i:i+1])).sample(seed=key, sample_shape=(num_samples)) for i in range(model.num_latents)])# [S, n, N]
        log_probs = new_model.likelihood.link_function(samples_f).log_prob(D.y[-num_test:,:].T) # [N, S]
        return jnp.mean(log_probs)


    thinned_model = model
    kept_idxs = [[i for i in range(len(base_kernels))] for base_kernels in model.base_kernels]
    for _ in range(len(model.base_kernels[0])-target_num):
        for j in range(model.num_latents):
            scores = jnp.array([test_model_without_component(D, thinned_model, [j,i]) for i in kept_idxs[j]])
            #print(scores)
            chosen_idx = jnp.argmax(scores)
            actual_idx = kept_idxs[j][chosen_idx]
            del kept_idxs[j][chosen_idx]
            thinned_model = test_model_without_component(D, thinned_model, [j, actual_idx], return_model=True)
            print(f"removed {problem_info.names_short[actual_idx]} from latent {j}")
    
    return thinned_model



def optim_builder(optim_pytree):

    def _init_leaf(o, p):
        if isinstance(o, GradientTransformation):
            return o.init(p)
        else:
            return None

    def _update_leaf(o, u, s, p):
        if isinstance(o, GradientTransformation):
            return tuple(o.update(u, s, p))
        else:
            return jtu.tree_map(jnp.zeros_like, p)

    def _get_updates(o, u, p):
        if isinstance(o, GradientTransformation):
            return u[0]
        else:
            return u
    
    def _get_state(o, u):
        if isinstance(o, GradientTransformation):
            return u[1]
        else:
            return None

    def init_fn(params):
        return jtu.tree_map(_init_leaf, optim_pytree, params, is_leaf=lambda x: isinstance(x, GradientTransformation))

    def update_fn(updates, state, params):
        updates_state = jtu.tree_map(_update_leaf, optim_pytree, updates, state, params, is_leaf=lambda x: isinstance(x, GradientTransformation))
        updates = jtu.tree_map(_get_updates, optim_pytree, updates_state, params, is_leaf=lambda x: isinstance(x, GradientTransformation))
        state = jtu.tree_map(_get_state, optim_pytree, updates_state, is_leaf=lambda x: isinstance(x, GradientTransformation))

        return updates, state

    return GradientTransformation(init_fn, update_fn)
