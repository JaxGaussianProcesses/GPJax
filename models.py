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
class VariationalPrecipGP(Module):
    base_kernels: List[List[gpx.kernels.AbstractKernel]]
    likelihood: gpx.likelihoods.AbstractLikelihood
    variational_mean: Union[Float[Array, "L N 1"], None] = param_field(None)
    variational_root_covariance: Float[Array, "L N N"] = param_field(
        None, bijector=tfb.FillTriangular()
    )
    num_latents: int = static_field(1)
    jitter: float = static_field(1e-6)
    parameterisation: str = static_field("standard") # "standard" or "white"
    inducing_inputs: Float[Array, "N D"] = param_field(None)
    max_interaction_depth: bool = static_field(2)
    interaction_variances: Float[Array, "L D"] = param_field(jnp.array([[1.0,1.0,1.0]]), bijector=tfb.Softplus(low=jnp.array(1e-5, dtype=jnp.float64)))
    jitter: float = static_field(1e-6)
    measure:str = static_field("empirical")
    use_shared_kernel: bool=static_field(True)
    dists: dict = static_field(None)
    ref: gpx.Dataset = static_field(None)
    
    def __post_init__(self):
        self.mean_function = gpx.mean_functions.Zero()
        if not self.max_interaction_depth == len(self.interaction_variances[0]) - 1:
            raise ValueError("Number of interaction variances must be equal to max_interaction_depth + 1")
        assert jnp.shape(self.variational_mean)[0] == self.num_latents
        assert jnp.shape(self.variational_root_covariance)[0] == self.num_latents
        if self.use_shared_kernel:
            assert len(self.base_kernels) == 1
        else:
            assert len(self.base_kernels) == self.num_latents
        for kernels in self.base_kernels:
            for k in kernels:
                assert isinstance(k, Union[gpx.kernels.RBF, gpx.kernels.Constant])
        

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
      

    def _get_ref(self):
        if self.ref is None:
            return self.get_inducing_locations()
        else:
            return self.ref.X


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
            Kzz = self.eval_K_xt(z, z, ref=self._get_ref()) # [L, M M]
            Kzz = Kzz + jnp.eye(self.num_inducing)[None,:,:] * self.jitter# [L, M M]
            pus = [tfd.MultivariateNormalFullCovariance(loc=muz[j], covariance_matrix=Kzz[j]) for j in range(self.num_latents)]# [L, M M]

        kl = jnp.stack([qus[j].kl_divergence(pus[j]) for j in range(self.num_latents)]) # [L]
        return jnp.sum(kl) # [1]
   

    def _predict(
        self,
        test_inputs: Num[Array, "N D"],
        component_list: Optional[List[List[int]]]=None,
    ) -> Union[GaussianDistribution,  Num[Array, "N 1"]]:
        r"""Get the posterior predictive distribution."""

        
        # Unpack variational parameters
        mu = self.variational_mean # [L, M, 1]
        sqrt = self.variational_root_covariance # [L, M, M]
        z = self.get_inducing_locations()# [N, d]
        
        Kzz = self.eval_K_xt(z, z, ref=self._get_ref()) # [L, M, M]
        Kzz = Kzz + jnp.eye(jnp.shape(z)[0])[None,:,:] * self.jitter # [L, M, M]
        Lz = jnp.linalg.cholesky(Kzz) # [L, M, M]
        muz = self.mean_function(z)[None,:,:] # [1, M, 1]


        # Unpack test inputs
        t = test_inputs
        if component_list is None:
            Ktt = self.eval_K_xt(t,t, ref=self._get_ref()) # [L, N, N]
            Kzt = self.eval_K_xt(z,t, ref=self._get_ref()) # [L, M, N]
        else:
            Ktt = self.eval_specific_K_xt(t,t, component_list, ref = self._get_ref()) # [L, N, N]
            Kzt = self.eval_specific_K_xt(z,t, component_list, ref = self._get_ref()) # [L, N, N]
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


    def eval_K_xt(self, x: Num[Array, "N d"], t: Num[Array, "M d"], ref:  Num[Array, "n d"]) -> Num[Array, "L N M"]:
        if self.measure == "empirical":
            x_all, t_all = jnp.vstack([x, ref]), jnp.vstack([t, ref])  # [N+n, d] [M+n, d]
            ks_all = jnp.stack([jnp.stack([k.cross_covariance(x_all,t_all) for k in kernels]) for kernels in self.base_kernels]) # [L, d, N+n, M+n]
            ks_all = self._orthogonalise_empirical(ks_all, num_ref = jnp.shape(ref)[0])  # [L, d, N, M]
        elif self.measure == "Gaussian":
            ks_all = jnp.stack([jnp.stack([k.cross_covariance(x,t) for k in kernels]) for kernels in self.base_kernels]) # [L, d, N, M]
            ks_all = self._orthogonalise_gaussian(ks_all, x, t) # [L, d, N, M]
        elif self.measure is None:
            ks_all = jnp.stack([jnp.stack([k.cross_covariance(x,t) for k in kernels]) for kernels in self.base_kernels]) # [L, d, N, M]
        else:
            raise ValueError("measure must be empirical, uniform or None")
        k = jnp.sum(self._compute_additive_terms_girad_newton(ks_all) * self.interaction_variances[:,:, None, None], 1) # [L, N, M]
        if self.use_shared_kernel:
            k = jnp.tile(k, (self.num_latents, 1, 1))
        return k



    def eval_specific_K_xt(self, x: Num[Array, "N d"], t: Num[Array, "M d"], component_list: List[int], ref =  Num[Array, "n d"])-> Num[Array, "L N M"]:
        
        if len(component_list) == 0:
            return self.interaction_variances[:,0, None,None] * jnp.ones((self.num_latents, jnp.shape(x)[0], jnp.shape(t)[0])) # [L N M]
        
        if self.measure == "empirical":
            x_all, t_all = jnp.vstack([x, ref]), jnp.vstack([t, ref])  # [N+n, d] [M+n, d]
            ks_all = jnp.stack([jnp.stack([kernels[i].cross_covariance(x_all,t_all) for i in component_list]) for kernels in self.base_kernels]) # [L, p, N+n, M+n]
            ks_all = self._orthogonalise_empirical(ks_all, num_ref = jnp.shape(ref)[0]) # [p, N, M] 
        elif self.measure == "Gaussian":
            ks_all = jnp.stack([jnp.stack([k.cross_covariance(x,t) for k in kernels]) for kernels in self.base_kernels]) # [L, d, N, M]
            ks_all = self._orthogonalise_gaussian(ks_all, x, t) # [L, d, N, M]
            ks_all = ks_all[:,component_list,:,:] # [L, p, N, M
        elif self.measure is None:
            ks_all = jnp.stack([jnp.stack([kernels[i].cross_covariance(x,t) for i in component_list]) for kernels in self.base_kernels]) # [L, p, N, M]
        else:
            raise ValueError("measure must be empirical, uniform or None")
        k = self.interaction_variances[:,len(component_list), None, None] * jnp.prod(ks_all,1) # [L N, M] 
        if self.use_shared_kernel:
            k = jnp.tile(k, (self.num_latents, 1, 1))
        return k



    def predict_indiv(
        self,
        test_inputs: Num[Array, "N D"],
        component_list: Optional[List[List[int]]]=None,
    ):
        def q_moments(x):
            qx = self._predict(x, component_list=component_list)
            return qx[0], qx[1]
        mean, var = vmap(q_moments)(test_inputs[:, None,:]) 
        return mean[:,:,0,0], var[:,:,0,0]






    def get_sobol_indicies(self, data:gpx.Dataset, component_list: List[List[int]], greedy=True, use_range=False, num_samples=100) -> Num[Array, "L c"]:

        if not isinstance(component_list, List):
            raise ValueError("Use get_sobol_index if you want to calc for single components (TODO)")


        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        x = data.X[-num_samples:,:]
        z = self.get_inducing_locations()
        m_z = jnp.tile(self.mean_function(z)[None,:,:], (self.num_latents, 1, 1))
        m_x = jnp.tile(self.mean_function(x)[None,:,:], (self.num_latents, 1, 1))


        if self.measure == "empirical":
            ref = self._get_ref()
            x_all = jnp.vstack([x,ref]) # waste of memory here
            z_all = jnp.vstack([z,ref]) # waste of memory here
            Kxz_indiv = jnp.stack([jnp.stack([k.cross_covariance(x_all,z_all) for k in kernels], axis=0) for kernels in self.base_kernels])  # [L, d, N + M, M + M]
            Kxz_indiv =  self._orthogonalise_empirical(Kxz_indiv, num_ref = jnp.shape(ref)[0]) # [L, d, N, M]
        elif self.measure == "Gaussian":
            Kxz_indiv = jnp.stack([jnp.stack([k.cross_covariance(x, z) for k in kernels]) for kernels in self.base_kernels]) # [L, p, N, M]
            Kxz_indiv  = self._orthogonalise_gaussian(Kxz_indiv , x,z) # [L, d, N, M]
        elif self.measure is None:
            Kxz_indiv = jnp.stack([jnp.stack([k.cross_covariance(x,z) for k in kernels], axis=0) for kernels in self.base_kernels])  # [L, d, N, M]
        else:
            raise ValueError("measure must be empirical, uniform or None")
        
        
        Kxz_components = [self.interaction_variances[:,len(c),None,None]*jnp.prod(Kxz_indiv[:,c, :, :], axis=1) for c in component_list] 
        Kxz_components = jnp.stack(Kxz_components, axis=0) # [c,L, N, N]
        Kxz_components = jnp.transpose(Kxz_components, (1,0,2,3)) # [L, c, N, N]
    
        
        if self.use_shared_kernel:
            Kxz_components = jnp.tile(Kxz_components, (self.num_latents, 1, 1))
        assert Kxz_components.shape[1] == len(component_list)


        Kzz = self.eval_K_xt(z, z, ref =  self._get_ref()) # [L, M, M]
        Kzz =Kzz + jnp.eye(self.num_inducing)[None,:,:] * self.jitter
        Kxz = self.eval_K_xt(x,z, ref =  self._get_ref())
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
        
        # print("NOTE THAT WE ARE CLIPPING BELOW 0 for SOBOL")
        # mean_overall = jnp.clip(mean_overall, a_min=1e-6)
        # mean_components = jnp.clip(mean_components, a_min=1e-6) 
        if greedy:
            raise ValueError("greedy doesnt work")
        
        
        if use_range:
            sobols = jnp.max(mean_components[:,:,:,0], axis=-1) - jnp.min(mean_components[:,:,:,0], axis=-1) # [L, c]  
        else:
            sobols = jnp.var(mean_components[:,:,:,0], axis=-1) / jnp.var(mean_overall[:,:,0], axis=-1, keepdims=True) # [L, c]  
        return sobols
    
    
    def _orthogonalise_empirical(self, ks: Num[Array, "L d N+n M+n"], num_ref: int)->Num[Array, "L d N M"]:
        ks_xt, ks_xX, ks_Xt, ks_XX = ks[:,:,:-num_ref,:-num_ref], ks[:,:,:-num_ref,-num_ref:], ks[:,:,-num_ref:,:-num_ref], ks[:,:,-num_ref:,-num_ref:] # [L, d, N, M], [L, d, N, n], [L, d, n, M], [L, d, n, n]
        denom = jnp.mean(ks_XX, (2,3))[:,:, None, None]+self.jitter # [L, d, 1, 1]
        Kx =  jnp.mean(ks_xX, 3) # [L, d, N]
        Kt = jnp.mean(ks_Xt, 2) # [L, d, M]
        numerator = jnp.matmul(Kx[:, :,:,None], Kt[:, :, None, :])# [L, d, N, M]
        return ks_xt -  numerator / denom 

    def _orthogonalise_gaussian(self, ks: Num[Array, "L d N M"],x: Num[Array, "N d"], t: Num[Array, "M d"])->Num[Array, "L d N M"]:
            l2 =jnp.stack([jnp.array([k.lengthscale[0] if isinstance(k, gpx.kernels.RBF) else 0.0 for k in kernels ]) for kernels in self.base_kernels])[:,:,None,None]**2 # [L, d, 1 , 1]
            cov_x_s = jnp.sqrt(l2 / (l2 + 1.0)) * jnp.exp(-0.5 * (x.T ** 2)[None, :,:,None] / (l2 + 1.0)) # [L, d, N, 1]
            cov_t_s = jnp.sqrt(l2 / (l2 + 1.0)) * jnp.exp(-0.5 * (t.T ** 2)[None, :,None,:] / (l2 + 1.0)) # [L, d, 1, M]
            var_s = jnp.clip(jnp.sqrt(l2 / (l2 + 2.0)), a_min=1e-5) # [L, d, 1, 1]
            return   ks -  cov_x_s * cov_t_s / var_s # [L, d, N, M]
        


    @jax.jit   
    def _compute_additive_terms_girad_newton(self, ks: Num[Array, "L D N M"]) -> Num[Array, "L p N M"]:
        L = jnp.shape(ks)[0]
        N = jnp.shape(ks)[-2]
        M = jnp.shape(ks)[-1]
        powers = jnp.arange(self.max_interaction_depth + 1)[:, None] # [p + 1, 1]
        s = jnp.power(ks[:, None, :,:,:],powers[None, :,:,None,None]) # [L, p + 1, d, N,M]
        e = jnp.ones(shape=(L,self.max_interaction_depth+1, N, M), dtype=jnp.float64) # [L, p+1, N, M]lazy init
        for n in range(1, self.max_interaction_depth + 1): # has to be for loop because iterative
            thing = jax.vmap(lambda k: ((-1.0)**(k-1))*e[:,n-k]*jnp.sum(s[:,k], 1))(jnp.arange(1, n+1))
            e = e.at[:,n].set((1.0/n) *jnp.sum(thing,0))
        return e

