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

from gpjax.precip_gp import VerticalSmoother, VerticalDataset, ProblemInfo


class ApproxPrecipGP(Module):
    pass



@dataclass
class VariationalPrecipGP(ApproxPrecipGP):
    
    list_of_list_of_base_kernels:List[List[gpx.kernels.AbstractKernel]]
    likelihood: gpx.likelihoods.AbstractLikelihood
    smoother: VerticalSmoother
    variational_mean: Union[Float[Array, "L N 1"], None] = param_field(None)
    variational_root_covariance: Float[Array, "L N N"] = param_field(
        None, bijector=tfb.FillTriangular()
    )
    num_latents: int = static_field(1)
    max_interaction_depth: bool = static_field(2)
    interaction_variances: Float[Array, "1 D"] = param_field(jnp.array([[1.0,1.0,1.0]]), bijector=tfb.Softplus(low=jnp.array(1e-5, dtype=jnp.float64)))
    jitter: float = static_field(1e-6)
    measure:str = static_field("empirical")
    inducing_inputs_3d: Float[Array, "N D L"] = param_field(None)
    inducing_inputs_2d: Float[Array, "N D"] = param_field(None)
    inducing_inputs_static: Float[Array, "N D"] = param_field(None)
    parameterisation: str = static_field("standard") # "standard" or "white"
    D_ref: Optional[VerticalDataset] = static_field(None)
    fiddled_ng: Optional[bool] = static_field(False)
    lengthscale_penalisation_factor: Optional[float] = static_field(1.0)
    variance_penalisation_factor: Optional[float] = static_field(1.0)
    

    def __post_init__(self):
        self.mean_function = gpx.mean_functions.Zero()
        self.num_latents = len(self.list_of_list_of_base_kernels)
        assert self.interaction_variances.shape[0] == self.num_latents
        assert jnp.shape(self.variational_mean)[0] == self.num_latents
        assert jnp.shape(self.variational_root_covariance)[0] == self.num_latents
        
        if self.fiddled_ng:
            assert self.max_interaction_depth == 2
            assert self.measure is None
            assert isinstance(self.list_of_list_of_base_kernels[0][0], gpx.kernels.RBF)

        if not self.max_interaction_depth == jnp.shape(self.interaction_variances)[1] - 1:
            raise ValueError("Number of interaction variances must be equal to max_interaction_depth + 1")


        assert self.measure in ["empirical", None]
        assert self.parameterisation in ["standard", "white"]
   

    def __call__(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        raise NotImplementedError
    
    @property
    def num_inducing(self) -> int:
        """The number of inducing inputs."""
        return self.inducing_inputs_3d.shape[0]


    def get_ref(self):
        if self.D_ref is None:
            return self.get_inducing_locations()
        else:
            return self.smoother.smooth_data(self.D_ref)[0]

    def get_inducing_locations(self):
        #return self.inducing_inputs
        smoothed = self.smoother.smooth_X(self.inducing_inputs_3d)
        return jnp.hstack([smoothed, self.inducing_inputs_2d, self.inducing_inputs_static])


    def _orthogonalise_empirical(self, ks: Num[Array, "L d N+n M+n"], num_ref: int)->Num[Array, "L d N M"]:
        ks_xt, ks_xX, ks_Xt, ks_XX = ks[:,:,:-num_ref,:-num_ref], ks[:,:,:-num_ref,-num_ref:], ks[:,:,-num_ref:,:-num_ref], ks[:,:,-num_ref:,-num_ref:] # [L, d, N, M], [L, d, N, n], [L, d, n, M], [L, d, n, n]
        denom = jnp.mean(ks_XX, (2,3))[:,:, None, None]+self.jitter # [L, d, 1, 1]
        Kx =  jnp.mean(ks_xX, 3) # [L, d, N]
        Kt = jnp.mean(ks_Xt, 2) # [L, d, M]
        numerator = jnp.matmul(Kx[:, :,:,None], Kt[:, :, None, :])# [L, d, N, M]
        return ks_xt -  numerator / denom 





    @jax.jit   
    def _compute_additive_terms_girad_newton(self, ks: Num[Array, "L D N M"]) -> Num[Array, "L p N M"]:
        N = jnp.shape(ks)[-2]
        M = jnp.shape(ks)[-1]
        powers = jnp.arange(self.max_interaction_depth + 1)[:, None] # [p + 1, 1]
        s = jnp.power(ks[:, None, :,:,:],powers[None, :,:,None,None]) # [L, p + 1, d, 1,1]
        e = jnp.ones(shape=(self.num_latents,self.max_interaction_depth+1, N, M), dtype=jnp.float64) # [L, p+1, N, M]lazy init then populate
        for n in range(1, self.max_interaction_depth + 1): # has to be for loop because iterative
            thing = jax.vmap(lambda k: ((-1.0)**(k-1))*e[:,n-k]*jnp.sum(s[:,k], 1))(jnp.arange(1, n+1))
            e = e.at[:,n].set((1.0/n) *jnp.sum(thing,0))
        return e



    def eval_K_xt(self, x: Num[Array, "N d"], t: Num[Array, "M d"], ref:  Num[Array, "n d"]) -> Num[Array, "L N M"]:
        if self.measure == "empirical":
            x_all, t_all = jnp.vstack([x, ref]), jnp.vstack([t, ref])  # [N+n, d] [M+n, d]
            ks_all = jnp.stack([jnp.stack([k.cross_covariance(x_all,t_all) for k in base_kernels]) for base_kernels in self.list_of_list_of_base_kernels]) # [L, d, N+n, M+n]
            ks_all = self._orthogonalise_empirical(ks_all, num_ref = jnp.shape(ref)[0])  # [L, d, N, M]
        elif self.measure is None:
            ks_all = jnp.stack([jnp.stack([k.cross_covariance(x,t) for k in base_kernels]) for base_kernels in self.list_of_list_of_base_kernels]) # [L, d, N, M]
        else:
            raise ValueError("measure must be empirical, uniform or None")
        if not self.fiddled_ng:
            return jnp.sum(self._compute_additive_terms_girad_newton(ks_all) * self.interaction_variances[:, :, None, None], 1) # [L N M]
        else:
            ng_1 = self._compute_additive_terms_girad_newton(ks_all) # [L p N M]
            ng_2 = self._compute_additive_terms_girad_newton(jnp.power(ks_all, 1/(2*self.lengthscale_penalisation_factor))) # [L p N M]
            return jnp.sum(ng_1[:,:-1,:,:] * self.interaction_variances[:, :-1, None, None] + ng_2[:,-1:,:,:] * self.interaction_variances[:, -1:, None, None],1) # [L N M]
            
            


    def eval_specific_K_xt(self, x: Num[Array, "N d"], t: Num[Array, "M d"], component_list: List[int], ref =  Num[Array, "n d"])-> Num[Array, "L N M"]:
        
        if len(component_list) == 0:
            return self.interaction_variances[:,0, None,None] * jnp.ones((self.num_latents, jnp.shape(x)[0], jnp.shape(t)[0])) # [L N M]
        
        if self.measure == "empirical":
            x_all, t_all = jnp.vstack([x, ref]), jnp.vstack([t, ref])  # [N+n, d] [M+n, d]
            ks_all = jnp.stack([jnp.stack([base_kernels[i].cross_covariance(x_all,t_all) for i in component_list]) for base_kernels in self.list_of_list_of_base_kernels]) # [L, p, N+n, M+n]
            ks_all = self._orthogonalise_empirical(ks_all, num_ref = jnp.shape(ref)[0]) # [L, p, N, M]
                
        elif self.measure is None:
            ks_all = jnp.stack([jnp.stack([base_kernels[i].cross_covariance(x,t) for i in component_list]) for base_kernels in self.list_of_list_of_base_kernels]) # [L, d, N, M]
        else:
            raise ValueError("measure must be empirical, uniform or None")
  
        if not self.fiddled_ng:
            return self.interaction_variances[:,len(component_list), None, None] * jnp.prod(ks_all,1) # [L, N, M] 
        else:
            return self.interaction_variances[:,len(component_list), None, None] * jnp.prod(jnp.power(ks_all, 1/(len(component_list)*self.lengthscale_penalisation_factor)),1) # [L, N, M] 


    def predict_indiv(
        self,
        test_inputs: Num[Array, "N D"],
        component_list: Optional[List[List[int]]]=None,
    ):
        def q_moments(x):
            qx = self._predict(x, component_list=component_list)
            return qx[0], qx[1]
        mean, var = vmap(q_moments)(test_inputs[:, None,:]) 
        return mean[:,:,0,0].T, var[:,:,0,0].T


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
            Kzz = self.eval_K_xt(z, z, ref = self.get_ref()) # [L, M M]
            Kzz = Kzz + jnp.eye(self.num_inducing)[None,:,:] * self.jitter# [L, M M]
            pus = [tfd.MultivariateNormalFullCovariance(loc=muz[j], covariance_matrix=Kzz[j]) for j in range(self.num_latents)]# [L, M M]

        kl = jnp.stack([qus[j].kl_divergence(pus[j]) for j in range(self.num_latents)]) # [L]
        return jnp.sum(kl) # [1]
   


    def _predict(
        self,
        test_inputs: Num[Array, "N D"],
        component_list: Optional[List[List[int]]]=None,
    ) -> Union[GaussianDistribution,  Num[Array, "N 1"]]:
        r"""Get the posterior predictive distribution (for a specific additive component if componen specified)."""

        
        # Unpack variational parameters
        mu = self.variational_mean # [L, M, 1]
        sqrt = self.variational_root_covariance # [L, M, M]
        z = self.get_inducing_locations()# [N, d]
        
        Kzz = self.eval_K_xt(z, z, ref =  self.get_ref()) # [L, M, M]
        Kzz = Kzz + jnp.eye(jnp.shape(z)[0])[None,:,:] * self.jitter # [L, M, M]
        Lz = jnp.linalg.cholesky(Kzz) # [L, M, M]
        muz = self.mean_function(z)[None,:,:] # [1, M, 1]


        # Unpack test inputs
        t = test_inputs
        if component_list is None:
            Ktt = self.eval_K_xt(t,t,ref= self.get_ref()) # [L, N, N]
            Kzt = self.eval_K_xt(z,t, ref= self.get_ref()) # [L, M, N]
        else:
            Ktt = self.eval_specific_K_xt(t, t, component_list, ref =  self.get_ref()) # [L, N, N]
            Kzt = self.eval_specific_K_xt(z, t, component_list, ref =  self.get_ref())   # [L, M, N]
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

    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        component_list: Optional[List[List[int]]]=None,
    ) -> Union[GaussianDistribution,  Num[Array, "N 1"]]:
        raise NotImplementedError


    def loss_fn(self, negative=False)->gpx.objectives.AbstractObjective:
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
                return self.constant * (elbo.squeeze())
        return Loss(negative=negative)
    
    
    def _custom_variational_expectation(
        self,
        model: ApproxPrecipGP,
        train_data: VerticalDataset,
    ) -> Float[Array, " N"]:
        # Unpack training batch
        x,y = model.smoother.smooth_data(train_data)
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
    


    
    
    def get_sobol_indicies(self, train_data: Optional[VerticalDataset], component_list: List[List[int]], use_inducing_points=False, use_ref=False, greedy=True) -> Num[Array, "L c"]:
        
        
        
        if not isinstance(component_list, List):
            raise ValueError("Use get_sobol_index if you want to calc for single components (TODO)")
        assert not ( use_inducing_points and use_ref)
        if use_inducing_points:
            x = self.get_inducing_locations()
        if use_ref:
            x = self.get_ref()
        else:
            x,y = self.smoother.smooth_data(train_data)
            
        
        
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        z = self.get_inducing_locations()
        m_x = self.mean_function(x)
        m_z = self.mean_function(z)
        


        if self.measure == "empirical":
            x_all = jnp.vstack([x,z]) # waste of memory here
            z_all = jnp.vstack([z,z]) # waste of memory here
            Kxz_indiv = jnp.stack([jnp.stack([k.cross_covariance(x_all,z_all) for k in base_kernels], axis=0) for base_kernels in self.list_of_list_of_base_kernels]) # [L, d, N + M, M + M]
            Kxz_indiv =  self._orthogonalise_empirical(Kxz_indiv, num_ref = jnp.shape(z)[0]) # [L, d, N, M]
        elif self.measure is None:
            Kxz_indiv = jnp.stack([jnp.stack([k.cross_covariance(x,z) for k in base_kernels], axis=0) for base_kernels in self.list_of_list_of_base_kernels]) # [d, N, M]
        else:
            raise ValueError("measure must be empirical, uniform or None")
        Kxz_components = [self.interaction_variances[:,len(c),None,None]*jnp.prod(Kxz_indiv[:,c, :, :], axis=1) for c in component_list] 
        Kxz_components = jnp.stack(Kxz_components, axis=0) # [c, L, N, N]
        Kxz_components = jnp.transpose(Kxz_components, (1,0,2,3)) # [L, c, N, N]
        if self.fiddled_ng:
            factors = jnp.array([len(c) for c in component_list])[None,:,None,None] # [c, 1, 1, 1]
            factors = factors * self.lengthscale_penalisation_factor
            Kxz_components = jnp.power(Kxz_components, 1/factors)
        assert Kxz_components.shape[1] == len(component_list)


        Kzz = self.eval_K_xt(z, z, ref =  self.get_ref())
        Kzz =Kzz + jnp.eye(self.num_inducing)[None,:,:] * self.jitter
        Kxz = self.eval_K_xt(x,z, ref =  self.get_ref())
        Lz = jnp.linalg.cholesky(Kzz)

        if self.parameterisation == "white":
            def get_mean_from_covar(K): # [L,N,N] -> [L,N, 1]
                Lz_inv_Kzt = jnp.linalg.solve(Lz, jnp.transpose(K, (0,2,1)))
                return m_x + jnp.matmul(jnp.transpose(Lz_inv_Kzt, (0,2,1)), mu)
        else:
            def get_mean_from_covar(K): # [L,N,N] -> [LN, 1]
                Lz_inv_Kzt = jnp.linalg.solve(Lz, jnp.transpose(K, (0,2,1)))
                Kzz_inv_Kzt = jnp.linalg.solve(jnp.transpose(Lz, (0,2,1)), Lz_inv_Kzt)
                return m_x + jnp.matmul(jnp.transpose(Kzz_inv_Kzt, (0,2,1)), mu - m_z)
            

        mean_overall =  get_mean_from_covar(Kxz) # [L,N, 1]
        mean_components = jnp.transpose(vmap(get_mean_from_covar)(jnp.transpose(Kxz_components,(1,0,2,3))), (1,0,2,3)) # [L,c, N, 1]


        # mean_overall = self.likelihood.link_function(mean_overall).mean()
        # mean_components = self.likelihood.link_function(mean_components).mean()


        if not greedy:
            sobols = jnp.var(mean_components[:,:,:,0], axis=-1) / jnp.var(mean_overall[:,:,0]) # [L, c]
        else:
            sobols = jnp.zeros((self.num_latents,len(component_list)))
            cumulative_pred = jnp.zeros((self.num_latents, jnp.shape(x)[0])) # [L, N]
            for i in range(len(component_list)):
                # v_12 = jnp.var(mean_components[:,:,:,0] - cumulative_pred[:,None,:], axis=-1)  # [L, c]
                # v_1 = jnp.var(mean_components[:,:,:,0],-1)# [L, c]
                # v_2 = jnp.var(cumulative_pred[:,None,:],-1)# [L, 1]
                # cov = 0.5 * (v_1 + v_2 - v_12) # [L, c]
                # current_sobols = v_1 - cov**2/(v_2+1e-6)
                current_sobols = jnp.var(mean_components[:,:,:,0] + cumulative_pred[:,None,:], axis=-1)  # [L, c]

                max_idx = jnp.argmax(current_sobols,1) #[L]
                for j in range(self.num_latents):
                    scores = current_sobols[j][max_idx[j]]
                    sobols = sobols.at[j,max_idx[j]].set(len(component_list) - i)
                cumulative_pred += jnp.stack([mean_components[j,max_idx[j],:,0] for j in range(self.num_latents)]) # [L, N]
                for j in range(self.num_latents):
                    mean_components = mean_components.at[j,max_idx[j]].set(jnp.zeros((jnp.shape(x)[0],1)))
        return sobols
    

    
    
@dataclass  
class VariationalPrecipGPSample(Module):
    model: VariationalPrecipGP = None
    key: jr.PRNGKey = static_field(jr.PRNGKey(123))
    
    
    def __post_init__(self):
        assert self.model.parameterisation == "standard"
        
        
    def predict(self, test_inputs: Num[Array, "N D"], component=List[int], num_samples=1) -> Num[Array, "N 1"]:
        mean, cov = self.model._predict(test_inputs, component_list=component)
        L = jnp.linalg.cholesky(cov) #+ jnp.eye(jnp.shape(test_inputs)[0]) * self.model.jitter)
        return mean + jnp.matmul(L, jr.normal(self.key, (self.model.num_latents,jnp.shape(test_inputs)[0], num_samples)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def thin_model(problem_info:ProblemInfo, D_test:VerticalDataset, model:VariationalPrecipGP, target_num):
    def test_model_without_component(D: VerticalDataset, model:VariationalPrecipGP, idx:List[int], return_model = False, num_samples=100):
        list_of_list_of_base_kernels = []
        for j in range(model.num_latents):
            new_base_kernels = []
            for i in range(len(model.list_of_list_of_base_kernels[j])):
                if i == idx[1] and j == idx[0]:
                    new_base_kernels.append(gpx.kernels.Constant(constant=jnp.array(0.0, dtype=jnp.float64), active_dims=[i]))
                else:
                    new_base_kernels.append(model.list_of_list_of_base_kernels[j][i])
                    
            list_of_list_of_base_kernels.append(new_base_kernels)
        new_model = model.replace(list_of_list_of_base_kernels=list_of_list_of_base_kernels)#VariationalPrecipGP(
        if return_model:
            return new_model 
        test_inputs = new_model.smoother.smooth_data(D)[0]
        mean, var = new_model.predict_indiv(test_inputs)
        samples_f = jnp.stack([tfd.MultivariateNormalDiag(mean[i], jnp.sqrt(var[i])).sample(seed=key, sample_shape=(num_samples)) for i in range(model.num_latents)])# [S, n, N]
        log_probs = new_model.likelihood.link_function(samples_f).log_prob(D.y.T) # [N, S]
        return jnp.mean(log_probs)


    thinned_model = model
    kept_idxs = [[i for i in range(len(base_kernels))] for base_kernels in model.list_of_list_of_base_kernels]
    for _ in range(len(model.list_of_list_of_base_kernels[0])-target_num):
        for j in range(model.num_latents):
            scores = jnp.array([test_model_without_component(D_test, thinned_model, [j,i]) for i in kept_idxs[j]])
            #print(scores)
            chosen_idx = jnp.argmax(scores)
            actual_idx = kept_idxs[j][chosen_idx]
            del kept_idxs[j][chosen_idx]
            thinned_model = test_model_without_component(D_test, thinned_model, [j, actual_idx], return_model=True)
            print(f"removed {problem_info.names_short[actual_idx]} from latent {j}")
    
    return thinned_model



def init_smoother(problem_info:ProblemInfo):
    smoother_input_scale_bijector = tfb.Softplus(low=jnp.array(0.1, dtype=jnp.float64))
    smoother_mean_bijector =  tfb.SoftClip(low=jnp.min(problem_info.pressure_levels), high=jnp.max(problem_info.pressure_levels))
    smoother = VerticalSmoother(
        jnp.array([[0.0]*problem_info.num_3d_variables]), 
        jnp.array([[1.0]*problem_info.num_3d_variables]), 
        Z_levels=problem_info.pressure_levels
        ).replace_bijector(smoother_input_scale=smoother_input_scale_bijector,smoother_mean=smoother_mean_bijector)
    return smoother


def init_kernels(problem_info:ProblemInfo):
    lengthscale_bij = tfb.SoftClip(low=jnp.array(1e-2, dtype=jnp.float64), high=jnp.array(1e2, dtype=jnp.float64))
    kernels = [] 
    for i, name in enumerate(problem_info.names_short):
        kernel = gpx.kernels.RBF(lengthscale=jnp.array([1.1]), active_dims=[i]).replace_trainable(variance=False).replace_bijector(lengthscale = lengthscale_bij)
        kernels.append(kernel)
    return kernels
    