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
    





@dataclass
class VerticalDataset(Pytree):
    X3d: Num[Array, "N D L"] = None
    X2d: Num[Array, "N D"] = None
    Xstatic: Num[Array, "N D"] = None
    y: Num[Array, "N 1"] = None
    standardize: bool = True
    standardize_with_NF: bool = False
    Y_mean: Num[Array, "1 1"] = None
    Y_std: Num[Array, "1 1"] = None
    problem_info: ProblemInfo = None
    

    def __post_init__(self) -> None:
        _check_precision(self.X2d, self.y)
        _check_precision(self.Xstatic, self.y)
        _check_precision(self.X3d, self.y)
        self.X3d_raw = self.X3d
        if self.standardize_with_NF:
            assert self.standardize
        
        
        if self.standardize:
            if self.standardize_with_NF:
                print("standardized X2d and Xstatic inputs with NF")
                def fit_normaliser(data, num_transforms):
                    d = jnp.shape(data)[1]
                    for i in range(num_transforms):
                        normaliser = gpx.normalizer.Normalizer(x=data, sinharcsinh_skewness=jnp.array([0.0]*d), sinharcsinh_tailweight=jnp.array([1.0]*d), standardizer_scale=jnp.array([1.0/jnp.std(data,0)]), standardizer_shift=jnp.array([-jnp.mean(data,0)]))
                        opt_normaliser, history = gpx.fit_scipy(model = normaliser, objective = normaliser.loss_fn(negative=False), train_data = None, safe=False)
                        data = opt_normaliser.get_bijector()(data)
                    return data
                
                X2d, Xstatic = self.X2d, self.Xstatic
                num_transforms = 1

                for i in range(len(self.problem_info.names_2d)):
                    print(f"standardizing {self.problem_info.names_2d[i]}")
                    X2d = X2d.at[:,i:i+1].set(fit_normaliser(X2d[:,i:i+1], num_transforms))
                

                # for i in range(len(self.problem_info.names_static)):
                #     print(f"standardizing {self.problem_info.names_static[i]}")
                #     #if self.problem_info.names_static[i]=="Land-sea Mask"
                #     Xstatic = Xstatic.at[:,i:i+1].set(fit_normaliser(Xstatic[:,i:i+1], num_transforms))
            else:
                X2d = self.X2d
                Xstatic = self.Xstatic

            print("Standardized X2d and Xstatic with max and min")
            lower = 0.001
            upper = 0.999
            X2d_min = jnp.quantile(X2d, lower, axis=0)
            X2d_max = jnp.quantile(X2d,upper, axis=0)
            X2d = jnp.clip(X2d, X2d_min, X2d_max)
            X2d = (X2d - X2d_min) / (X2d_max - X2d_min)
            Xstatic_min = jnp.quantile(Xstatic,lower, axis=0)
            Xstatic_max = jnp.quantile(Xstatic,upper, axis=0)
            Xstatic = jnp.clip(Xstatic, Xstatic_min, Xstatic_max)
            Xstatic = (Xstatic - Xstatic_min) / (Xstatic_max - Xstatic_min)
            
            # print("robust scale of X2d and gaussian of Xstatic")
            # X2d_median = jnp.median(self.X2d, axis=0)
            # X2d_lower_quartile = jnp.percentile(self.X2d, 25, axis=0)
            # X2d_upper_quartile = jnp.percentile(self.X2d, 75, axis=0)
            # X2d = (self.X2d - X2d_median) / (X2d_upper_quartile - X2d_lower_quartile)

                        
            

            print(" overall standardized X3d with max and min")
            X3d = self.X3d
            X3d = (self.X3d - jnp.mean(self.X3d, 0)) / jnp.std(self.X3d, 0)
            X3d_max = jnp.max(X3d,axis=(0,2))
            X3d_min = jnp.min(X3d, axis=(0,2))
            X3d = (X3d-X3d_min[None,:,None]) / (X3d_max[None,:,None] - X3d_min[None,:,None])
               
               
               
            print(f"no Y standarisation")
            y=self.y
            self.Y_mean = 0.0
            self.Y_std = 1.0
            
            self.X3d = X3d
            self.X2d = X2d
            self.Xstatic = Xstatic
            self.y = y
            

    @property
    def X(self):
        return NotImplementedError("Use X2d, X3d or Xstatic instead")

    @property
    def n(self):
        return self.X2d.shape[0]

    @property
    def dim(self):
        return self.X2d.shape[1] + self.X3d.shape[1] + self.Xstatic.shape[1]



    def get_subset(self, M: int, space_filling=False,use_output=False, no_3d=True, smoother=None):
        if space_filling:
            if use_output:
                if no_3d:
                    X = jnp.hstack([self.X2d, self.Xstatic, self.y])
                else:
                    #X = jnp.hstack([jnp.mean(self.X3d,-1), self.X2d, self.Xstatic, self.y])
                    X = jnp.hstack([jnp.reshape(self.X3d,(jnp.shape(self.X3d)[0],-1)), self.X2d, self.Xstatic, self.y])
            else:
                if no_3d:
                    X = jnp.hstack([self.X2d, self.Xstatic])
                else:
                    if smoother is not None:
                        smoothed = smoother.smooth_X(self.X3d)
                    X = jnp.hstack([smoothed, self.X2d, self.Xstatic])
                    #X = jnp.hstack([jnp.reshape(self.X3d,(jnp.shape(self.X3d)[0],-1)), self.X2d, self.Xstatic])
            assert X.shape[0] > M
            d = X.shape[1]
            #kernel = gpx.kernels.SumKernel(kernels=[gpx.kernels.RBF(active_dims=[i], lengthscale=jnp.array(1.0 , dtype=jnp.float64)) for i in range(d)])
            kernel = gpx.kernels.RBF(lengthscale=jnp.array(0.1, dtype=jnp.float64))
            chosen_indicies = []  # iteratively store chosen points
            N = X.shape[0]
            c = jnp.zeros((M - 1, N), dtype=jnp.float64)  # [M-1,N]
            d_squared = vmap(lambda x: kernel(x,x),0)(X) # [N]

            chosen_indicies.append(jnp.argmax(d_squared))  # get first element
            for m in range(M - 1):  # get remaining elements
                ix = jnp.array(chosen_indicies[-1], dtype=int) # increment Cholesky with newest point
                newest_point = X[ix]
                d_temp = jnp.sqrt(d_squared[ix])  # [1]

                L = kernel.cross_covariance(X, newest_point[None, :])[:, 0]  # [N]
                if m == 0:
                    e = L / (d_temp+1e-3)
                    c = e[None,:]  # [1,N]
                else:
                    c_temp = c[:, ix : ix + 1]  # [m,1]
                    e = (L - jnp.matmul(jnp.transpose(c_temp), c[:m])) / (d_temp+1e-3)  # [N]
                    c = jnp.concatenate([c, e], axis=0)  # [m+1, N]
                    # e = tf.squeeze(e, 0)
                d_squared -= e**2
                d_squared = jnp.maximum(d_squared, 1e-50)  # numerical stability
                #print(d_squared)
                chosen_indicies.append(jnp.argmax(d_squared))  # get next element as point with largest score
            chosen_indicies = jnp.array(chosen_indicies, dtype=int)
            if jnp.all(d_squared==1e-50):
                print("space filling probs didnt work")
            print(d_squared)
            return VerticalDataset(X3d=self.X3d[chosen_indicies], X2d=self.X2d[chosen_indicies], Xstatic=self.Xstatic[chosen_indicies], y=self.y[chosen_indicies], standardize=False, Y_mean=self.Y_mean, Y_std=self.Y_std)
        else:
            return VerticalDataset(X3d=self.X3d[:M], X2d=self.X2d[:M], Xstatic=self.Xstatic[:M], y=self.y[:M], standardize=False, Y_mean=self.Y_mean, Y_std=self.Y_std)



