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


from utils import VerticalDataset

@dataclass
class VerticalSmoother(Module):
    smoother_mean: Float[Array, "1 D"]  = param_field(None)
    smoother_input_scale: Float[Array, "1 D"] = param_field(None, bijector=tfb.Softplus())
    Z_levels: Float[Array, "1 L"] = static_field(jnp.array([[1.0]]))
    eps: float = static_field(1e-6)
    
    def smooth_fn(self, x):
        #return jnp.exp(-0.5 * ((x - self.smoother_mean.T) / self.smoother_input_scale.T) ** 2)
        lower = jax.scipy.stats.norm.cdf(jnp.min(self.Z_levels)-1e-3, loc = self.smoother_mean.T, scale = self.smoother_input_scale.T)
        upper = jax.scipy.stats.norm.cdf(jnp.max(self.Z_levels)+1e-3, loc = self.smoother_mean.T, scale = self.smoother_input_scale.T)
        return  jax.scipy.stats.norm.pdf(x, self.smoother_mean.T, self.smoother_input_scale.T) / (upper - lower)
        #return  jax.scipy.stats.truncnorm.pdf(x, a = jnp.min(self.Z_levels)-0.1, b = jnp.max(self.Z_levels)+0.1, loc = self.smoother_mean.T, scale = self.smoother_input_scale.T)

    def smooth(self) -> Num[Array, "D L"]:
        return self.smooth_fn(self.Z_levels)
        
    
    def smooth_data(self, dataset: VerticalDataset, batch_size: Optional[int]=None) -> VerticalDataset:
        x3d, x2d, xstatic, y = dataset.X3d, dataset.X2d, dataset.Xstatic, dataset.y
        x3d_smooth = self.smooth_X(x3d)
        x = jnp.hstack([x3d_smooth, x2d, xstatic]) # [N, D_3d + D_2d +D_static]
        return x, y

    def smooth_X(self, X3d):
        delta = self.Z_levels[:,1:] - self.Z_levels[:,:-1] # [1 L-1]
        weights=self.smooth()[:,:-1] # [D L]
        x3d_smooth = jnp.sum(jnp.multiply(jnp.multiply(weights[None,:,:],delta[None,:,:]) , X3d[:,:,:-1]), axis=-1) # [N, D_3d]
        #x3d_smooth = x3d_smooth / (jnp.max(self.Z_levels) - jnp.min(self.Z_levels))
        #x3d_smooth = (x3d_smooth - jnp.min(x3d_smooth, axis=0)) / (jnp.max(x3d_smooth, axis=0) - jnp.min(x3d_smooth, axis=0))
        #x3d_smooth = (x3d_smooth - jnp.mean(x3d_smooth, axis=0)) / jnp.sqrt(jnp.var(x3d_smooth, axis=0)+ self.eps)
        return x3d_smooth        
    

def init_smoother(problem_info):
    smoother_input_scale_bijector = tfb.Softplus(low=jnp.array(0.1, dtype=jnp.float64))
    smoother_mean_bijector =  tfb.SoftClip(low=jnp.min(problem_info.pressure_levels), high=jnp.max(problem_info.pressure_levels))
    smoother = VerticalSmoother(
        jnp.array([[0.0]*problem_info.num_3d_variables]), 
        jnp.array([[1.0]*problem_info.num_3d_variables]), 
        Z_levels=problem_info.pressure_levels
        ).replace_bijector(smoother_input_scale=smoother_input_scale_bijector,smoother_mean=smoother_mean_bijector)
    return smoother

@dataclass
class ConjugatePrecipGP(Module):
    kernel:gpx.kernels.AbstractKernel
    likelihood: gpx.likelihoods.AbstractLikelihood
    smoother: VerticalSmoother
    jitter: float = static_field(1e-6)
    
    def __post_init__(self):
        self.mean_function = gpx.mean_functions.Zero()


        
    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: VerticalDataset,
    ) -> Union[GaussianDistribution,  Num[Array, "N 1"]]:
        r"""Get the posterior predictive distribution."""
        #smooth data to get in form for preds
        x, y = self.smoother.smooth_data(train_data)
        t = test_inputs
        
        obs_noise = self.likelihood.obs_stddev**2
        mx = self.mean_function(x)
        mt = self.mean_function(t)
        
        
        Kxx = self.kernel.gram(x) # [N, N]
        Ktt = self.kernel.gram(t) # [M, M]
        Kxt = self.kernel.cross_covariance(x,t) # [N, M]
        Kxx += cola.ops.I_like(Kxx) * self.jitter
        Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
        Sigma = cola.PSD(Sigma)
        Sigma_inv_Kxt = cola.solve(Sigma, Kxt, Cholesky())
        mean = mt + jnp.matmul(Sigma_inv_Kxt.T, y - mx)
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += cola.ops.I_like(covariance) * self.jitter
        covariance = cola.PSD(covariance)
        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)
    


    def loss_fn(self, negative=False)->gpx.objectives.AbstractObjective:
        class Loss_mll(gpx.objectives.AbstractObjective):
            def step(
                self,
                posterior: ConjugatePrecipGP,
                train_data: gpx.Dataset,
            ) -> ScalarFloat:
                #smooth data to get in form for preds
                x, y = posterior.smoother.smooth_data(train_data)
                
                obs_noise = posterior.likelihood.obs_stddev**2
                mx = posterior.mean_function(x)
                Kxx = posterior.kernel.gram(x) # [N, N]
                
                # Σ = (Kxx + Io²) = LLᵀ
                Kxx += cola.ops.I_like(Kxx) * posterior.jitter
                Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
                Sigma = cola.PSD(Sigma)

                # p(y | x, θ), where θ are the model hyperparameters:
                mll = GaussianDistribution(jnp.atleast_1d(mx.squeeze()), Sigma)
                return self.constant * (mll.log_prob(jnp.atleast_1d(y.squeeze())).squeeze())
            

        return Loss_mll(negative=negative)