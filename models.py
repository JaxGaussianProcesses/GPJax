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
class ConjugatePrecipGP(Module):
    base_kernels:List[gpx.kernels.AbstractKernel]
    likelihood: gpx.likelihoods.AbstractLikelihood
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
        train_data: gpx.Dataset,
        component_list: Optional[List[List[int]]]=None,
    ) -> Union[GaussianDistribution,  Num[Array, "N 1"]]:
        r"""Get the posterior predictive distribution (for a specific additive component if componen specified)."""
        #smooth data to get in form for preds
        x, y = train_data.X, train_data.y
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
    
    def loss_fn(self, negative=False)->gpx.objectives.AbstractObjective:
        class Loss_mll(gpx.objectives.AbstractObjective):
            def step(
                self,
                posterior: ConjugatePrecipGP,
                train_data: gpx.Dataset,
            ) -> ScalarFloat:
                #smooth data to get in form for preds
                x, y = train_data.X, train_data.y
                
                obs_noise = posterior.likelihood.obs_stddev**2
                mx = posterior.mean_function(x)
                Kxx = posterior.eval_K_xt(x,x, ref=x) # [N, N]
                
                # Σ = (Kxx + Io²) = LLᵀ
                Kxx += cola.ops.I_like(Kxx) * posterior.jitter
                Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
                Sigma = cola.PSD(Sigma)

                # p(y | x, θ), where θ are the model hyperparameters:
                mll = GaussianDistribution(jnp.atleast_1d(mx.squeeze()), Sigma)

                log_prob = 0.0
                lengthscales = jnp.array([k.lengthscale for k in posterior.base_kernels])
                log_prob += jnp.sum(tfd.LogNormal(loc=0.0, scale=1.0).log_prob(lengthscales))
                log_prob += jnp.sum(tfd.LogNormal(loc=0.0, scale=1.0).log_prob(posterior.interaction_variances))
                return self.constant * (mll.log_prob(jnp.atleast_1d(y.squeeze())).squeeze() + log_prob.squeeze())
            

        return Loss_mll(negative=negative)
    


    def predict_indiv_mean(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: gpx.Dataset,
        component_list: Optional[List[List[int]]]=None,
    ):
        predictor = lambda x: self.predict(x, train_data, component_list).mean()
        return jax.vmap(predictor,1)(test_inputs[:,None,:]).T
    
    def predict_indiv_var(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: gpx.Dataset,
        component_list: Optional[List[List[int]]]=None,
    ):
        predictor = lambda x: self.predict(x, train_data, component_list).variance()
        return jax.vmap(predictor,1)(test_inputs[:,None,:]).T


    def get_sobol_indicies(self, train_data: gpx.Dataset, component_list: List[List[int]], use_range=False, greedy=False) -> Num[Array, "c"]:

        
        if not isinstance(component_list, List):
            raise ValueError("Use get_sobol_index if you want to calc for single components (TODO)")
        x,y = train_data.X, train_data.y
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
        
        if greedy:
            sobols = jnp.zeros((len(component_list)))
            cumulative_pred = jnp.zeros((jnp.shape(x)[0])) # [ N]
            for i in range(len(component_list)):
                if use_range:
                    current_sobols = jnp.max(mean_components[:,:,0] - cumulative_pred[None,:], axis=-1) - jnp.min(mean_components[:,:,0] - cumulative_pred[None,:], axis=-1)
                else:
                    current_sobols = jnp.var(mean_components[:,:,0] - cumulative_pred[None,:], axis=-1)  / jnp.var(mean_overall - cumulative_pred) # [L, c]

                max_idx = jnp.argmax(current_sobols)
                sobols = sobols.at[max_idx].set(len(component_list) - i)
                cumulative_pred += mean_components[max_idx,:,0] # [N]
                mean_components = mean_components.at[max_idx].set(jnp.zeros((jnp.shape(x)[0],1)))
       

        return sobols
        
        
        
        
        
        
        


@dataclass
class VariationalPrecipGP(ConjugatePrecipGP):
    base_kernels:List[gpx.kernels.AbstractKernel]
    likelihood: gpx.likelihoods.AbstractLikelihood
    variational_mean: Union[Float[Array, "L N 1"], None] = param_field(None)
    variational_root_covariance: Float[Array, "L N N"] = param_field(
        None, bijector=tfb.FillTriangular()
    )
    num_latents: int = static_field(1)
    jitter: float = static_field(1e-6)
    parameterisation: str = static_field("standard") # "standard" or "white"
    inducing_inputs: Float[Array, "N D"] = param_field(None)

    

    def __post_init__(self):
        self.mean_function = gpx.mean_functions.Zero()
        assert jnp.shape(self.variational_mean)[0] == self.num_latents
        assert jnp.shape(self.variational_root_covariance)[0] == self.num_latents



    def predict(
        self,
        test_inputs: Num[Array, "N D"],
    ) -> Union[GaussianDistribution,  Num[Array, "N 1"]]:
        raise NotImplementedError
    
    @property
    def num_inducing(self) -> int:
        """The number of inducing inputs."""
        return self.inducing_inputs.shape[0]

    def get_inducing_locations(self):
        return self.inducing_inputs



    def loss_fn(self, negative=False)->gpx.objectives.AbstractObjective:
        class Loss(gpx.objectives.AbstractObjective):
            def step(
                self,
                model: VariationalPrecipGP,
                train_data: gpx.Dataset,
            ) -> ScalarFloat:
                elbo = (
                    jnp.sum(jnp.sum(model._custom_variational_expectation(model, train_data)))
                    * model.likelihood.num_datapoints
                    / train_data.n
                    - model.prior_kl()
                ) 
                return self.constant * (elbo.squeeze())
        return Loss(negative=negative)
    

    def _custom_variational_expectation(
        self,
        model: Module,
        train_data: gpx.Dataset,
    ) -> Float[Array, " N"]:
        # Unpack training batch
        x,y =train_data.X, train_data.y
        q = model
        def q_moments(x):
            qx = q._predict(x)
            return qx[0], qx[1]
        mean, variance = vmap(q_moments)(x[:, None,:]) # [N, L, 1, 1], [N, L, 1, 1]
        mean = mean[:,:,0,0].T # [L N]
        variance = variance[:,:,0,0].T # [L N]
        expectation = q.likelihood.expected_log_likelihood(
            y, mean, variance
        )
        return expectation
      


    def prior_kl(self) -> ScalarFloat:
        # Unpack variational parameters
        mu = self.variational_mean # [L, M, 1]
        sqrt = self.variational_root_covariance # [L, M, M]
        z = self.get_inducing_locations() # [N, d]
        
        # S = LLᵀApprox
        S = sqrt @ jnp.transpose(sqrt,(0,2,1)) + jnp.eye(self.num_inducing)[None,:,:] * self.jitter # [L M N]
        qus  = [tfd.MultivariateNormalFullCovariance(loc=mu[i],covariance_matrix=S[i]) for i in range(self.num_latents)]
        if self.parameterisation == "white":
            pus = [tfd.MultivariateNormalFullCovariance(loc=jnp.zeros_like(mu[j]),covariance_matrix=jnp.eye(self.num_inducing)) for j in range(self.num_latents)]
        else:
            muz = self.mean_function(z)[None, :, :] # [1, M, 1]
            Kzz = self.eval_K_xt(z, z, ref=z) # [L, M M]
            Kzz = Kzz + jnp.eye(self.num_inducing)[None,:,:] * self.jitter# [L, M M]
            pus = [tfd.MultivariateNormalFullCovariance(loc=muz[j], covariance_matrix=Kzz[j]) for j in range(self.num_latents)]# [L, M M]

        kl = jnp.stack([qus[j].kl_divergence(pus[j]) for j in range(self.num_latents)]) # [L]
        return jnp.sum(kl) # [1]
   

    def _predict(
        self,
        test_inputs: Num[Array, "N D"],
    ) -> Union[GaussianDistribution,  Num[Array, "N 1"]]:
        r"""Get the posterior predictive distribution."""

        
        # Unpack variational parameters
        mu = self.variational_mean # [L, M, 1]
        sqrt = self.variational_root_covariance # [L, M, M]
        z = self.get_inducing_locations()# [N, d]
        
        Kzz = self.eval_K_xt(z, z, ref=z) # [L, M, M]
        Kzz = Kzz + jnp.eye(jnp.shape(z)[0])[None,:,:] * self.jitter # [L, M, M]
        Lz = jnp.linalg.cholesky(Kzz) # [L, M, M]
        muz = self.mean_function(z)[None,:,:] # [1, M, 1]


        # Unpack test inputs
        t = test_inputs
        Ktt = self.eval_K_xt(t,t, ref=z) # [L, N, N]
        Kzt = self.eval_K_xt(z,t,ref=z) # [L, M, N]
        mut = self.mean_function(t)[None,:,:] # [1, M, 1]

        if self.parameterisation == "white":
            # Lz⁻¹ Kzt
            Lz_inv_Kzt = jax.scipy.linalg.solve(Lz, Kzt)#cola.solve(Lz, Kzt, Cholesky()) TODO CHECK THIS

            # Ktz Lz⁻ᵀ sqrt
            Ktz_Lz_invT_sqrt = jnp.matmul(jnp.transpose(Lz_inv_Kzt, (0,2,1)), sqrt)

            # μt  +  Ktz Lz⁻ᵀ μ
            mean = mut + jnp.matmul(jnp.transpose(Lz_inv_Kzt, (0,2,1)), mu)

            # Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Lz⁻ᵀ S Lz⁻¹ Kzt  [recall S = sqrt sqrtᵀ]
            covariance = (
                Ktt
                - jnp.matmul(jnp.transpose(Lz_inv_Kzt, (0,2,1)), Lz_inv_Kzt)
                + jnp.matmul(Ktz_Lz_invT_sqrt, jnp.transpose(Ktz_Lz_invT_sqrt, (0,2,1)))
            )

        else:
            # Lz⁻¹ Kzt
            Lz_inv_Kzt = jax.scipy.linalg.solve(Lz, Kzt) # [L M N]

            # Kzz⁻¹ Kzt
            Kzz_inv_Kzt = jax.scipy.linalg.solve(jnp.transpose(Lz,(0,2,1)), Lz_inv_Kzt)

            # Ktz Kzz⁻¹ sqrt
            Ktz_Kzz_inv_sqrt = jnp.matmul(jnp.transpose(Kzz_inv_Kzt, (0,2,1)), sqrt)

            # μt + Ktz Kzz⁻¹ (μ - μz)
            mean = mut + jnp.matmul(jnp.transpose(Kzz_inv_Kzt, (0,2,1)), mu - muz)

            # Ktt - Ktz Kzz⁻¹ Kzt  +  Ktz Kzz⁻¹ S Kzz⁻¹ Kzt  [recall S = sqrt sqrtᵀ]
            covariance = (
                Ktt
                - jnp.matmul(jnp.transpose(Lz_inv_Kzt, (0,2,1)), Lz_inv_Kzt)
                + jnp.matmul(Ktz_Kzz_inv_sqrt, jnp.transpose(Ktz_Kzz_inv_sqrt, (0,2,1)))
            )


        covariance += jnp.eye(jnp.shape(t)[0])[None,:,:] * self.jitter
        return mean, covariance



