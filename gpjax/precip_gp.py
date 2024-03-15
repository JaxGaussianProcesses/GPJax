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


from jaxtyping import Num, Float
from simple_pytree import Pytree
from gpjax.typing import Array
import gpjax as gpx
from jax import vmap
import tensorflow_probability.substrates.jax.bijectors as tfb
from gpjax.base import Module, param_field,static_field
from gpjax.lower_cholesky import lower_cholesky


import jax
from dataclasses import dataclass
import jax.numpy as jnp
import jax.random as jr

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
                def fit_normaliser(data):
                    d = jnp.shape(data)[1]
                    normaliser = gpx.normalizer.Normalizer(x=data, sinharcsinh_skewness=jnp.array([0.0]*d), sinharcsinh_tailweight=jnp.array([1.0]*d), standardizer_scale=jnp.array([1.0/jnp.std(data,0)]), standardizer_shift=jnp.array([-jnp.mean(data,0)]))
                    opt_normaliser, history = gpx.fit_scipy(model = normaliser, objective = normaliser.loss_fn(negative=False), train_data = None, safe=False)
                    return opt_normaliser.get_bijector()(data)
                
                X2d, Xstatic = self.X2d, self.Xstatic
                for _ in range(2):
                    X2d = fit_normaliser(X2d)
                    Xstatic = fit_normaliser(Xstatic)
                    
                print("then standardized X3d as Gaussian")
                X3d_mean= jnp.mean(self.X3d,axis=(0,2))
                X3d_std= jnp.std(self.X3d, axis=(0,2))
                X3d = (self.X3d-X3d_mean[None,:,None]) / (X3d_std[None,:,None])
                
            else:
                print("Standardized X2d and Xstatic with max and min")
                X2d_min = jnp.min(self.X2d, axis=0)
                X2d_max = jnp.max(self.X2d,axis=0)
                X2d = (self.X2d - X2d_min) / (X2d_max - X2d_min)
                Xstatic_min = jnp.min(self.Xstatic, axis=0)
                Xstatic_max = jnp.max(self.Xstatic,axis=0)
                Xstatic = (self.Xstatic - Xstatic_min) / (Xstatic_max - Xstatic_min)
                
                # print("robust scale of X2d and gaussian of Xstatic")
                # X2d_median = jnp.median(self.X2d, axis=0)
                # X2d_lower_quartile = jnp.percentile(self.X2d, 25, axis=0)
                # X2d_upper_quartile = jnp.percentile(self.X2d, 75, axis=0)
                # X2d = (self.X2d - X2d_median) / (X2d_upper_quartile - X2d_lower_quartile)

                         
                
                
                
                print(" remove mean then overall standardized X3d with max and min")
                X3d = self.X3d
                X3d = (self.X3d - jnp.mean(self.X3d, 0)) / jnp.std(self.X3d, 0)
                X3d_max = jnp.max(X3d,axis=(0,2))
                X3d_min = jnp.min(X3d, axis=(0,2))
                X3d = (X3d-X3d_min[None,:,None]) / (X3d_max[None,:,None] - X3d_min[None,:,None])
                # X3d_max = jnp.max(self.X3d,axis=(0))
                # X3d_min = jnp.min(self.X3d, axis=(0))
                # X3d = (self.X3d-X3d_min[None,:,:]) / (X3d_max[None,:,:] - X3d_min[None,:,:])
                # X3d = X3d - jnp.mean(X3d, 0)
            

                # print("Standardized X2d and Xstatic as Gaussin")
                # X2d_mean = jnp.mean(self.X2d, axis=0)
                # X2d_std = jnp.std(self.X2d,axis=0)
                # X2d = (self.X2d - X2d_mean) / (X2d_std)
                # Xstatic_mean = jnp.mean(self.Xstatic, axis=0)
                # Xstatic_std = jnp.std(self.Xstatic,axis=0)
                # Xstatic = (self.Xstatic - Xstatic_mean) / (Xstatic_std)
                
                # # print("do nothing to 3d")
                # # X3d = self.X3d
                
                # print("then standardized X3d as Gaussian")
                # X3d = (self.X3d - jnp.mean(self.X3d, 0))
                # X3d_mean = jnp.mean(X3d,axis=(0,2))
                # X3d_std = jnp.std(X3d, axis=(0,2))
                # X3d = (X3d-X3d_mean[None,:,None]) / (X3d_std[None,:,None])
                
                # X3d_std = jnp.std(self.X3d, axis=(0))
                # X3d = (self.X3d-X3d_mean[None,:,:]) / (X3d_std[None,:,:])





                # print(f"then standardized Y as Gaussian")
                # self.Y_mean = jnp.mean(self.y,0)
                # self.Y_std = jnp.sqrt(jnp.var(self.y,0))
                # y = (self.y-self.Y_mean) / self.Y_std
            
                # print(f"then standardized Y with max and min")
                # self.Y_mean = None
                # self.Y_std = None
                # y = (self.y - jnp.min(self.y)) / (jnp.max(self.y) - jnp.min(self.y))
                
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



    def get_subset(self, M: int, space_filling=False,use_output=False, no_3d=True):
        if space_filling:
            if use_output:
                if no_3d:
                    X = jnp.hstack([self.X2d, self.Xstatic, self.y])
                else:
                    X = jnp.hstack([jnp.mean(self.X3d,-1), self.X2d, self.Xstatic, self.y])
                    #X = jnp.hstack([jnp.reshape(self.X3d,(jnp.shape(self.X3d)[0],-1)), self.X2d, self.Xstatic, self.y])
            else:
                if no_3d:
                    X = jnp.hstack([self.X2d, self.Xstatic])
                else:
                    X = jnp.hstack([jnp.mean(self.X3d,-1), self.X2d, self.Xstatic])
                    #X = jnp.hstack([jnp.reshape(self.X3d,(jnp.shape(self.X3d)[0],-1)), self.X2d, self.Xstatic])
            assert X.shape[0] > M
            d = X.shape[1]
            #kernel = gpx.kernels.SumKernel(kernels=[gpx.kernels.RBF(active_dims=[i], lengthscale=jnp.array(0.1 , dtype=jnp.float64)) for i in range(d)])
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
                    e = L / d_temp
                    c = e[None,:]  # [1,N]
                else:
                    c_temp = c[:, ix : ix + 1]  # [m,1]
                    e = (L - jnp.matmul(jnp.transpose(c_temp), c[:m])) / d_temp  # [N]
                    c = jnp.concatenate([c, e], axis=0)  # [m+1, N]
                    # e = tf.squeeze(e, 0)
                d_squared -= e**2
                d_squared = jnp.maximum(d_squared, 1e-50)  # numerical stability
                print(d_squared)
                chosen_indicies.append(jnp.nanargmax(d_squared))  # get next element as point with largest score
            chosen_indicies = jnp.array(chosen_indicies, dtype=int)
            return VerticalDataset(X3d=self.X3d[chosen_indicies], X2d=self.X2d[chosen_indicies], Xstatic=self.Xstatic[chosen_indicies], y=self.y[chosen_indicies], standardize=False, Y_mean=self.Y_mean, Y_std=self.Y_std)
        else:
            return VerticalDataset(X3d=self.X3d[:M], X2d=self.X2d[:M], Xstatic=self.Xstatic[:M], y=self.y[:M], standardize=False, Y_mean=self.Y_mean, Y_std=self.Y_std)



@dataclass
class VerticalSmoother(Module):
    smoother_mean: Float[Array, "1 D"]  = param_field(None)
    smoother_input_scale: Float[Array, "1 D"] = param_field(None, bijector=tfb.Softplus())
    Z_levels: Float[Array, "1 L"] = static_field(jnp.array([[1.0]]))
    eps: float = static_field(1e-6)
    
    def smooth_fn(self, x):
        #return jnp.exp(-0.5 * ((x - self.smoother_mean.T) / self.smoother_input_scale.T) ** 2)
        lower = jax.scipy.stats.norm.cdf(jnp.min(self.Z_levels)-0.01, loc = self.smoother_mean.T, scale = self.smoother_input_scale.T)
        upper = jax.scipy.stats.norm.cdf(jnp.max(self.Z_levels)+0.01, loc = self.smoother_mean.T, scale = self.smoother_input_scale.T)
        return  jax.scipy.stats.norm.pdf(x, self.smoother_mean.T, self.smoother_input_scale.T) / (upper - lower)
        #return  jax.scipy.stats.truncnorm.pdf(x, a = jnp.min(self.Z_levels)-0.1, b = jnp.max(self.Z_levels)+0.1, loc = self.smoother_mean.T, scale = self.smoother_input_scale.T)

    def smooth(self) -> Num[Array, "D L"]:
        return self.smooth_fn(self.Z_levels)
        
        #smoothing_weights = jnp.exp(-0.5*((self.Z_levels-self.smoother_mean.T)/(self.smoother_input_scale.T))**2) # [D, L]
        #return   (smoothing_weights/ jnp.sum(smoothing_weights, axis=-1, keepdims=True)) # [D, L]
        #return  (smoothing_weights/ jnp.sqrt(jnp.sum(smoothing_weights**2, axis=-1, keepdims=True))) # [D, L]
    
    
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




@dataclass
class ConjugatePrecipGP(Module):
    base_kernels:List[gpx.kernels.AbstractKernel]
    likelihood: gpx.likelihoods.AbstractLikelihood
    smoother: VerticalSmoother
    max_interaction_depth: bool = static_field(2)
    interaction_variances: Float[Array, " D"] = param_field(jnp.array([1.0,1.0,1.0]), bijector=tfb.Softplus(low=jnp.array(1e-5, dtype=jnp.float64)))
    jitter: float = static_field(1e-6)
    measure:str = static_field("empirical")
    second_order_empirical: bool = static_field(False)
    pairs: Optional[Num[Array, "d 2"]] = static_field(None) # lazy init
    
    def __post_init__(self):
        self.mean_function = gpx.mean_functions.Zero()
        if not self.max_interaction_depth == len(self.interaction_variances) - 1:
            raise ValueError("Number of interaction variances must be equal to max_interaction_depth + 1")

        if self.second_order_empirical:
            assert self.measure == "empirical"
            assert self.max_interaction_depth == 2
            pairs= []
            for i in range(len(self.base_kernels)):
                for j in range(i):
                    pairs.append([i,j])
            self.pairs = jnp.array(pairs)
        assert self.measure in ["empirical", None]
        

        
    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: VerticalDataset,
        component_list: Optional[List[List[int]]]=None,
    ) -> Union[GaussianDistribution,  Num[Array, "N 1"]]:
        r"""Get the posterior predictive distribution (for a specific additive component if componen specified)."""
        #smooth data to get in form for preds
        x, y = self.smoother.smooth_data(train_data)
        t = test_inputs
        
        obs_noise = self.likelihood.obs_stddev**2
        mx = self.mean_function(x)
        mt = self.mean_function(t)
        
        
        Kxx = self.eval_K_xt(x, x, ref = x) # [N, N]
        Kxx += cola.ops.I_like(Kxx) * self.jitter
        Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
        Sigma = cola.PSD(Sigma)
        
        if component_list is None:
            Ktt = self.eval_K_xt(t,t,ref=x)
            Kxt = self.eval_K_xt(x,t, ref=x)
        else:
            Ktt = self.eval_specific_K_xt(t, t, component_list, ref = x)
            Kxt = self.eval_specific_K_xt(x, t, component_list, ref = x)
        Sigma_inv_Kxt = cola.solve(Sigma, Kxt, Cholesky())
        
        mean = mt + jnp.matmul(Sigma_inv_Kxt.T, y - mx)
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += cola.ops.I_like(covariance) * self.jitter
        covariance = cola.PSD(covariance)
        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)
    

    def get_second_order(self, ks: Num[Array, "d N M"])->Num[Array, "d N M"]: # turn [d, N, M] into [d+(d**2-d)/2 N M]        
        ks_second = ks[None,:,:,:] * ks[:,None,:,:] # [d, d, N+n, M+n]
        ks_second = jax.vmap(lambda x: ks_second[x[0],x[1],:,:])(self.pairs) # [(d**2-d)/2, N+n, M+n]
        return jnp.vstack([ks, ks_second]) # [d + (d**2-d)/2, N+n, M+n]
        


    def eval_K_xt(self, x: Num[Array, "N d"], t: Num[Array, "M d"], ref:  Num[Array, "n d"]) -> Num[Array, "N M"]:
        if self.measure == "empirical":
            x_all, t_all = jnp.vstack([x, ref]), jnp.vstack([t, ref])  # [N+n, d] [M+n, d]
            ks_all = jnp.stack([k.cross_covariance(x_all,t_all) for k in self.base_kernels]) # [d, N+n, M+n]
            if self.second_order_empirical:
                ks_all = self.get_second_order(ks_all) # [d + (d**2-d)/2), N+n, M+n]
            ks_all = self._orthogonalise_empirical(ks_all, num_ref = jnp.shape(ref)[0])  # [d, N, M]
        elif self.measure is None:
            ks_all = jnp.stack([k.cross_covariance(x,t) for k in self.base_kernels]) # [d, N, M]
        else:
            raise ValueError("measure must be empirical, uniform or None")
        
        if self.second_order_empirical:
            return self.interaction_variances[0] + self.interaction_variances[1] * jnp.sum(ks_all[:jnp.shape(x)[1],:,:], 0) + self.interaction_variances[2] * jnp.sum(ks_all[jnp.shape(x)[1]:,:,:], 0)
        else:
            return jnp.sum(self._compute_additive_terms_girad_newton(ks_all) * self.interaction_variances[:, None, None], 0)

        
    def eval_specific_K_xt(self, x: Num[Array, "N d"], t: Num[Array, "M d"], component_list: List[int], ref =  Num[Array, "n d"])-> Num[Array, "N M"]:
        if len(component_list) == 0:
            return self.interaction_variances[0] * jnp.ones((jnp.shape(x)[0], jnp.shape(t)[0]))
        
        if self.measure == "empirical":
            x_all, t_all = jnp.vstack([x, ref]), jnp.vstack([t, ref])  # [N+n, d] [M+n, d]
            ks_all = jnp.stack([self.base_kernels[i].cross_covariance(x_all,t_all) for i in component_list]) # [p, N+n, M+n]
            if self.second_order_empirical: # do prod before orthog
                ks_all = jnp.prod(ks_all, 0 , keepdims=True) # [1, N+n, M+n]
            ks_all = self._orthogonalise_empirical(ks_all, num_ref = jnp.shape(ref)[0]) # [p, N, M]
                
        elif self.measure is None:
            ks_all = jnp.stack([self.base_kernels[i].cross_covariance(x,t) for i in component_list]) # [d, N, M]
        else:
            raise ValueError("measure must be empirical, uniform or None")
            
        return self.interaction_variances[len(component_list)] * jnp.prod(ks_all,0) # [N, M] 
       
       
       
       
    def _orthogonalise_empirical(self, ks: Num[Array, "d N+n M+n"], num_ref: int)->Num[Array, "d N M"]:
        ks_xt, ks_xX, ks_Xt, ks_XX = ks[:,:-num_ref,:-num_ref], ks[:,:-num_ref,-num_ref:], ks[:,-num_ref:,:-num_ref], ks[:,-num_ref:,-num_ref:] # [d, N, M], [d, N, n], [d, n, M], [d, n, n]
        denom = jnp.mean(ks_XX, (1,2))[:, None, None]+self.jitter # [d, 1, 1]
        Kx =  jnp.mean(ks_xX, 2) # [d, N]
        Kt = jnp.mean(ks_Xt, 1) # [d, M]
        numerator = jnp.matmul(Kx[:,:,None], Kt[:, None, :])# [d, N, M]
        
        return ks_xt -  numerator / denom 


    @jax.jit   
    def _compute_additive_terms_girad_newton(self, ks: Num[Array, "D N M"]) -> Num[Array, "p N M"]:
        N = jnp.shape(ks)[-2]
        M = jnp.shape(ks)[-1]
        powers = jnp.arange(self.max_interaction_depth + 1)[:, None] # [p + 1, 1]
        s = jnp.power(ks[None, :,:,:],powers[:,:,None,None]) # [p + 1, d, 1,1]
        e = jnp.ones(shape=(self.max_interaction_depth+1, N, M), dtype=jnp.float64) # [p+1, N, N]lazy init then populate
        for n in range(1, self.max_interaction_depth + 1): # has to be for loop because iterative
            thing = jax.vmap(lambda k: ((-1.0)**(k-1))*e[n-k]*jnp.sum(s[k], 0))(jnp.arange(1, n+1))
            e = e.at[n].set((1.0/n) *jnp.sum(thing,0))
        return e
    
    def loss_fn(self, negative=False, log_prior: Optional[Callable] = None, use_loocv=False)->gpx.objectives.AbstractObjective:
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
                Kxx = posterior.eval_K_xt(x,x, ref=x) # [N, N]
                
                # Σ = (Kxx + Io²) = LLᵀ
                Kxx += cola.ops.I_like(Kxx) * posterior.jitter
                Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
                Sigma = cola.PSD(Sigma)

                # p(y | x, θ), where θ are the model hyperparameters:
                mll = GaussianDistribution(jnp.atleast_1d(mx.squeeze()), Sigma)
                log_prob =jnp.array(0.0, dtype=jnp.float64)
                if log_prior is not None:
                    log_prob += log_prior(posterior)
                return self.constant * (mll.log_prob(jnp.atleast_1d(y.squeeze())).squeeze() + log_prob.squeeze())
            
        class Loss_loocv(gpx.objectives.AbstractObjective):
            def step(
                self,
                posterior: ConjugatePrecipGP,
                train_data: gpx.Dataset,
            ) -> ScalarFloat:
                #smooth data to get in form for preds
                x, y = posterior.smoother.smooth_data(train_data)
                
                obs_noise = posterior.likelihood.obs_stddev**2
                mx = posterior.mean_function(x)
                Kxx = posterior.eval_K_xt(x,x, ref=x) # [N, N]
                Kxx += cola.ops.I_like(Kxx) * posterior.jitter
                Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
                Sigma = cola.PSD(Sigma)  # [N, N]
                
                Sigma_inv_y = cola.solve(Sigma, y - mx, Cholesky())  # [N, 1]
                Sigma_inv_diag = cola.linalg.diag(cola.inv(Sigma, Cholesky()))[
                    :, None
                ]  # [N, 1]

                loocv_means = mx + (y - mx) - Sigma_inv_y / Sigma_inv_diag
                loocv_stds = jnp.sqrt(1.0 / Sigma_inv_diag)

                loocv_posterior = tfd.Normal(loc=loocv_means, scale=loocv_stds)
                loocv = jnp.sum(loocv_posterior.log_prob(y))                
                log_prob =jnp.array(0.0, dtype=jnp.float64)
                if log_prior is not None:
                    log_prob += log_prior(posterior)
                return self.constant * (loocv.squeeze() + log_prob.squeeze())
              
            
            
        if use_loocv:
            return Loss_loocv(negative=negative, log_prior=log_prior)
        else:
            return Loss_mll(negative=negative, log_prior=log_prior)




    def predict_indiv_mean(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: VerticalDataset,
        component_list: Optional[List[List[int]]]=None,
    ):
        predictor = lambda x: self.predict(x, train_data, component_list).mean()
        return jax.vmap(predictor,1)(test_inputs[:,None,:])
    
    def predict_indiv_var(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: VerticalDataset,
        component_list: Optional[List[List[int]]]=None,
    ):
        predictor = lambda x: self.predict(x, train_data, component_list).variance()
        return jax.vmap(predictor,1)(test_inputs[:,None,:])


    def get_sobol_indicies(self, train_data: VerticalDataset, component_list: List[List[int]], use_range=False) -> Num[Array, "c"]:
        if not isinstance(component_list, List):
            raise ValueError("Use get_sobol_index if you want to calc for single components (TODO)")
        x,y = self.smoother.smooth_data(train_data)
        m_x = self.mean_function(x)
        if self.second_order_empirical:
            x_all = jnp.vstack([x,x]) # waste of memory here
            Kxx_indiv = jnp.stack([k.cross_covariance(x_all,x_all) for k in self.base_kernels], axis=0) # [d, 2N, 2N]
            Kxx_components = [jnp.prod(Kxx_indiv[c, :, :],0) for c in component_list]  
            Kxx_components = jnp.stack(Kxx_components, axis=0) # [c, N, N]
            Kxx_components =  self._orthogonalise_empirical(Kxx_components, num_ref = jnp.shape(x)[0]) # [d, N, N]
            Kxx_components = [self.interaction_variances[len(c)]*Kxx_components[i, :, :] for i, c in enumerate(component_list)]
            Kxx_components = jnp.stack(Kxx_components, axis=0) # [c, N, N]
            
        else:
            if self.measure == "empirical":
                x_all = jnp.vstack([x,x]) # waste of memory here
                Kxx_indiv = jnp.stack([k.cross_covariance(x_all,x_all) for k in self.base_kernels], axis=0) # [d, 2N, 2N]
                Kxx_indiv =  self._orthogonalise_empirical(Kxx_indiv, num_ref = jnp.shape(x)[0]) # [d, N, N]
            elif self.measure is None:
                Kxx_indiv = jnp.stack([k.cross_covariance(x,x) for k in self.base_kernels], axis=0) # [d, N, N]
            else:
                raise ValueError("measure must be empirical, uniform or None")
            Kxx_components = [self.interaction_variances[len(c)]*jnp.prod(Kxx_indiv[c, :, :], axis=0) for c in component_list] 
            Kxx_components = jnp.stack(Kxx_components, axis=0) # [c, N, N]
        
        assert Kxx_components.shape[0] == len(component_list)

        Kxx = self.eval_K_xt(x,x, ref=x)
        Sigma = cola.PSD(Kxx + cola.ops.I_like(Kxx) * (self.likelihood.obs_stddev**2+self.jitter))

        def get_mean_from_covar(K): # [N,N] -> [N, 1]
            Sigma_inv_Kxx = cola.solve(Sigma, K)
            return m_x + jnp.matmul(Sigma_inv_Kxx.T, y - m_x) # [N, 1] 

        mean_overall =  get_mean_from_covar(Kxx) # [N, 1]
        mean_components = vmap(get_mean_from_covar)(Kxx_components) # [c, N, 1]

        if use_range:
            sobols = jnp.max(mean_components[:,:,0], axis=-1) - jnp.min(mean_components[:,:,0], axis=-1) # [c]
        else:
            sobols = jnp.var(mean_components[:,:,0], axis=-1) / jnp.var(mean_overall) # [c]
        return sobols
    
    
    
    


class ApproxPrecipGP(ConjugatePrecipGP):
    pass
    

@dataclass
class VariationalPrecipGP(ApproxPrecipGP):

    variational_mean: Union[Float[Array, "N 1"], None] = param_field(None)
    variational_root_covariance: Float[Array, "N N"] = param_field(
        None, bijector=tfb.FillTriangular()
    )
    inducing_inputs_3d: Float[Array, "N D L"] = param_field(None)
    inducing_inputs_2d: Float[Array, "N D"] = param_field(None)
    inducing_inputs_static: Float[Array, "N D"] = param_field(None)

    @property
    def num_inducing(self) -> int:
        """The number of inducing inputs."""
        return self.inducing_inputs_3d.shape[0]


    def _smoothed_inducing_points(self):
        z_smoothed_3d = self.inducing_inputs_3d#self.smoother.smooth_X(self.inducing_inputs_3d)
        return jnp.hstack([z_smoothed_3d,self.inducing_inputs_2d, self.inducing_inputs_static])
        

    def prior_kl(self) -> ScalarFloat:
        r"""Compute the prior KL divergence.

        Compute the KL-divergence between our variational approximation and the
        Gaussian process prior.

        For this variational family, we have
        ```math
        \begin{align}
        \operatorname{KL}[q(f(\cdot))\mid\mid p(\cdot)] & = \operatorname{KL}[q(u)\mid\mid p(u)]\\
        & = \operatorname{KL}[ \mathcal{N}(\mu, S) \mid\mid N(\mu z, \mathbf{K}_{zz}) ],
        \end{align}
        ```
        where $`u = f(z)`$ and $`z`$ are the inducing inputs.

        Returns
        -------
             ScalarFloat: The KL-divergence between our variational
                approximation and the GP prior.
        """
        # Unpack variational parameters
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        
        z = self._smoothed_inducing_points()
        
        # Unpack mean function and kernel
        mean_function = self.mean_function
        muz = mean_function(z)
        Kzz = self.eval_K_xt(z, z, ref = z) # [N, N]
        
        Kzz = cola.PSD(Kzz + cola.ops.I_like(Kzz) * self.jitter)

        sqrt = cola.ops.Triangular(sqrt)
        S = sqrt @ sqrt.T + jnp.eye(self.num_inducing) * self.jitter
        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.atleast_1d(muz.squeeze()), scale=Kzz)
        return qu.kl_divergence(pu)
   
   
    def __call__(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        return self.predict(*args, **kwargs)
    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        component_list: Optional[List[List[int]]]=None,
    ) -> Union[GaussianDistribution,  Num[Array, "N 1"]]:
        r"""Get the posterior predictive distribution (for a specific additive component if componen specified)."""

        
        # Unpack variational parameters
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        z = self._smoothed_inducing_points()
        
        Kzz = self.eval_K_xt(z, z, ref = z)
        Kzz = cola.PSD(Kzz + cola.ops.I_like(Kzz) * self.jitter)
        Lz = lower_cholesky(Kzz)
        muz = self.mean_function(z)


        # Unpack test inputs
        t = test_inputs
        if component_list is None:
            Ktt = self.eval_K_xt(t,t,ref=z)
            Kzt = self.eval_K_xt(z,t, ref=z)
        else:
            Ktt = self.eval_specific_K_xt(t, t, component_list, ref = z)
            Kzt = self.eval_specific_K_xt(z, t, component_list, ref = z)  
        mut = self.mean_function(t)
        
        # Lz⁻¹ Kzt
        Lz_inv_Kzt = cola.solve(Lz, Kzt, Cholesky())

        # Kzz⁻¹ Kzt
        Kzz_inv_Kzt = cola.solve(Lz.T, Lz_inv_Kzt, Cholesky())

        # Ktz Kzz⁻¹ sqrt
        Ktz_Kzz_inv_sqrt = jnp.matmul(Kzz_inv_Kzt.T, sqrt)

        # μt + Ktz Kzz⁻¹ (μ - μz)
        mean = mut + jnp.matmul(Kzz_inv_Kzt.T, mu - muz)

        # Ktt - Ktz Kzz⁻¹ Kzt  +  Ktz Kzz⁻¹ S Kzz⁻¹ Kzt  [recall S = sqrt sqrtᵀ]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Kzz_inv_sqrt, Ktz_Kzz_inv_sqrt.T)
        )
        covariance += cola.ops.I_like(covariance) * self.jitter
        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )

    def predict_indiv_mean(
        self,
        test_inputs: Num[Array, "N D"],
        component_list: Optional[List[List[int]]]=None,
    ):
        predictor = lambda x: self.predict(x,  component_list).mean()
        return jax.vmap(predictor,1)(test_inputs[:,None,:])
    
    def predict_indiv_var(
        self,
        test_inputs: Num[Array, "N D"],
        component_list: Optional[List[List[int]]]=None,
    ):
        predictor = lambda x: self.predict(x,  component_list).variance()
        return jax.vmap(predictor,1)(test_inputs[:,None,:])




    def loss_fn(self, negative=False, log_prior: Optional[Callable] = None)->gpx.objectives.AbstractObjective:
        class Loss(gpx.objectives.AbstractObjective):
            def step(
                self,
                model: ApproxPrecipGP,
                train_data: VerticalDataset,
            ) -> ScalarFloat:
                #smooth data to get in form for preds
                elbo = (
                    jnp.sum(model._custom_variational_expectation(model, train_data))
                    * model.likelihood.num_datapoints
                    / train_data.n
                    - model.prior_kl()
                ) 
                log_prob =jnp.array(0.0, dtype=jnp.float64)
                if log_prior is not None:
                    log_prob += log_prior(model)
                return self.constant * (elbo.squeeze() + log_prob.squeeze())
        return Loss(negative=negative, log_prior=log_prior)
    
    
    def _custom_variational_expectation(
        self,
        model: ApproxPrecipGP,
        train_data: VerticalDataset,
    ) -> Float[Array, " N"]:
        # Unpack training batch
        x,y = model.smoother.smooth_data(train_data)
        q = model
        def q_moments(x):
            qx = q(x)
            return qx.mean().squeeze(), qx.covariance().squeeze()
        mean, variance = vmap(q_moments)(x[:, None])
        expectation = q.likelihood.expected_log_likelihood(
            y, mean[:, None], variance[:, None]
        )
        return expectation
    
    
    
    
    
    def get_sobol_indicies(self, train_data: Optional[VerticalDataset], component_list: List[List[int]], use_range=False, use_inducing_points=False) -> Num[Array, "c"]:
        if not isinstance(component_list, List):
            raise ValueError("Use get_sobol_index if you want to calc for single components (TODO)")
        if use_inducing_points:
            x = self._smoothed_inducing_points()
        else:
            x,y = self.smoother.smooth_data(train_data)
        
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        z = self._smoothed_inducing_points()
        m_x = self.mean_function(x)
        m_z = self.mean_function(z)
        
        if self.second_order_empirical:
            x_all = jnp.vstack([x,z]) # waste of memory here
            z_all = jnp.vstack([z,z]) # waste of memory here
            Kxx_indiv = jnp.stack([k.cross_covariance(x_all,z_all) for k in self.base_kernels], axis=0) # [d, N+M, 2M]
            Kxx_components = [jnp.prod(Kxx_indiv[c, :, :],0) for c in component_list]  
            Kxx_components = jnp.stack(Kxx_components, axis=0) # [c, N, N]
            Kxx_components =  self._orthogonalise_empirical(Kxx_components, num_ref = jnp.shape(x)[0]) # [d, N, M]
            Kxx_components = [self.interaction_variances[len(c)]*Kxx_components[i, :, :] for i, c in enumerate(component_list)]
            Kxx_components = jnp.stack(Kxx_components, axis=0) # [c, N, M]
        else:
            if self.measure == "empirical":
                x_all = jnp.vstack([x,z]) # waste of memory here
                z_all = jnp.vstack([z,z]) # waste of memory here
                Kxz_indiv = jnp.stack([k.cross_covariance(x_all,z_all) for k in self.base_kernels], axis=0) # [d, N + M, M + M]
                Kxz_indiv =  self._orthogonalise_empirical(Kxz_indiv, num_ref = jnp.shape(z)[0]) # [d, N, M]
            elif self.measure is None:
                Kxz_indiv = jnp.stack([k.cross_covariance(x,z) for k in self.base_kernels], axis=0) # [d, N, M]
            else:
                raise ValueError("measure must be empirical, uniform or None")
            Kxz_components = [self.interaction_variances[len(c)]*jnp.prod(Kxz_indiv[c, :, :], axis=0) for c in component_list] 
            Kxz_components = jnp.stack(Kxz_components, axis=0) # [c, N, N]
        
        assert Kxz_components.shape[0] == len(component_list)




        Kzz = self.eval_K_xt(z, z, ref = z)
        Kzz = cola.PSD(Kzz + cola.ops.I_like(Kzz) * self.jitter)
        Kxz = self.eval_K_xt(x,z, ref = z)
        Lz = lower_cholesky(Kzz)

        def get_mean_from_covar(K): # [N,N] -> [N, 1]
            Lz_inv_Kzt = cola.solve(Lz, K.T, Cholesky())
            Kzz_inv_Kzt = cola.solve(Lz.T, Lz_inv_Kzt, Cholesky())
            return m_x + jnp.matmul(Kzz_inv_Kzt.T, mu - m_z)
        

        mean_overall =  get_mean_from_covar(Kxz) # [N, 1]
        mean_components = vmap(get_mean_from_covar)(Kxz_components) # [c, N, 1]

        if not isinstance(self.likelihood, gpx.likelihoods.Gaussian):
            mean_overall = self.likelihood.link_function(mean_overall).mean()
            mean_components = self.likelihood.link_function(mean_components).mean()


        if use_range:
            sobols = jnp.max(mean_components[:,:,0], axis=-1) - jnp.min(mean_components[:,:,0], axis=-1) # [c]
        else:
            sobols = jnp.var(mean_components[:,:,0], axis=-1) / jnp.var(mean_overall) # [c]
        return sobols
    
    


       
@dataclass
class SwitchKernelPositive(gpx.kernels.AbstractKernel):
    r"""The linear kernel."""
    threshold: float = static_field(0.5)
    name: str = "SwitchKernel"

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        x_yes = x[0] > self.threshold
        y_yes = y[0] > self.threshold
        return (x_yes*y_yes).squeeze()
    
    
@dataclass
class SwitchKernelNegative(gpx.kernels.AbstractKernel):
    r"""The linear kernel."""
    threshold: float = static_field(0.5)
    name: str = "SwitchKernel"

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        x_yes = x[0] < self.threshold
        y_yes = y[0] < self.threshold
        return (x_yes*y_yes).squeeze()
    
    