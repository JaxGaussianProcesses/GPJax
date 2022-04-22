import jax.numpy as jnp
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from typing import Optional, Dict, Callable
from chex import dataclass
from jax import vmap

import distrax

from gpjax.config import get_defaults
from gpjax.gps import Posterior
from gpjax.types import Array, Dataset
from gpjax.utils import I, concat_dictionaries
from gpjax.parameters import transform
from gpjax.kernels import gram, cross_covariance

from .quadrature import gauss_hermite_quadrature
from .variational import VariationalGaussian, VariationalFamily, VariationalPosterior

DEFAULT_JITTER = get_defaults()["jitter"]


########################################################
### Gaussian process (GP) notation used in the code: ###
########################################################

#   - x     for the train inputs
#   - y     for train labels
#   - t     for test inputs

#   - F     for the latent function modelled as a GP
#   - Fx    for the latent function, F, at train inputs, x
#   - Fmu   for the predictive mean of the latent function, F
#   - Fcov  for the predictive covariance of the latent function, F
#   - Fvar  for the predictive (diagonal) variance of the latent function, F

#   - nx    for the number of train inputs, x
#   - Kxx   for the kernel gram matrix at train inputs, x
#   - Lx    for lower cholesky decomposition at train inputs, x
#   - mx    for prior mean at train inputs, x

#   - nt    for number of test inputs, t
#   - Ktt   for gram matrix at train inputs, t
#   - Lt    for lower cholesky decomposition at test inputs, t
#   - mt    for prior mean at test inputs, t

#   - Ktx   for cross covariance between test inputs, t, and train inputs, x

# For sparse GPs:

#   - u for the inducing outputs
#   - z for inducing inputs
#   - nz for number of inducing inputs, z
#   - Kzz for gram matrix at inducing inputs, z
#   - Lz for lower cholesky decomposition at inducing inputs, z
#   - Kzx   for cross covariance between test inputs, z, and train inputs, x


# The variational Gaussian has two parameterations:
#   - q(u) = N[u; mu,  sqrt.sqrt^T], is the standard (whiten is false).
#   - q(u) = N[u; Lz.mu + mz,  (Lz.sqrt).(Lz.sqrt)^T], is the whitened parameterisation.
# for variational parameters, mu and sqrt, a vector and lower-triangular matrix respectively to optimise over.

def SVGP(posterior: Posterior, variational_family: VariationalFamily) -> VariationalPosterior:
    return _SVGP(likelihood = posterior.likelihood, 
                prior = posterior.prior,
                q = variational_family)

@dataclass
class _SVGP(VariationalPosterior):
    q: VariationalFamily = VariationalGaussian
    jitter: Optional[float] = DEFAULT_JITTER 
        
    def __post_init__(self):
        self.num_inducing = self.q.num_inducing

    @property
    def params(self) -> Dict:
        hyperparams = concat_dictionaries(self.prior.params,
                                          {"likelihood": self.likelihood.params,
                                          "q": self.q.params})
        return hyperparams

    def elbo(self, train_data: Dataset, transformations: Dict) -> Callable[[Array], Array]:

        def elbo_fn(params: Dict, batch: Dataset) -> Array:
            params = transform(params, transformations)
            kl = self.prior_kl(params)
            var_exp = self.variational_expectation(params, batch)

            return jnp.sum(var_exp) * train_data.n/batch.n - kl
        
        return elbo_fn
    
    # Compute KL divergence at inducing points, KL[q(u)||p(u)]: 
    def prior_kl(self, params: Dict) -> Array:
        mu = params["q"]["mu"]
        sqrt = params["q"]["sqrt"]
        nz = self.num_inducing

        qu = distrax.MultivariateNormalTri(mu.squeeze(), sqrt)
        
        if not self.q.whiten:
            z = params["q"]["inducing_inputs"]
            mz = self.prior.mean_function(z, params["mean_function"])
            Kzz = gram(self.prior.kernel, z, params["kernel"])
            Kzz += I(nz) * self.jitter
            Lz = cholesky(Kzz, lower=True)
            pu = distrax.MultivariateNormalTri(mz.squeeze(), Lz)
            
        else:
            pu = distrax.MultivariateNormalDiag(jnp.zeros(nz))
        
        return qu.kl_divergence(pu)


    # Compute expectation, ∫ log(p(y|F)) q(F) dF, through Gauss-Hermite quadrature:
    def variational_expectation(self, params: Dict, batch: Dataset) -> Array:
        x, y = batch.X, batch.y

        Fmu, Fvar = vmap(self.pred_moments, in_axes=(None, 0))(params, x[:,jnp.newaxis,:])

        # Get log(p(y|F)) function for current likelihood parameter values:
        def log_prob(F, y):
            return self.likelihood.link_function(F, params["likelihood"]).log_prob(y)

        return gauss_hermite_quadrature(log_prob, Fmu.squeeze(1), Fvar.squeeze(1), y=y)
    
    # Computes predictive moments for Gauss-Hermite quadrature:
    def pred_moments(self, params: Dict, test_inputs: Array) -> Array:
        mu = params["q"]["mu"]
        sqrt = params["q"]["sqrt"]
        
        # Cholesky decomposition at inducing inputs:
        z = params["q"]["inducing_inputs"]
        nz = self.num_inducing
        Kzz = gram(self.prior.kernel, z, params["kernel"])
        Kzz += I(nz) * self.jitter
        Lz = cholesky(Kzz)
        
        # Compute predictive moments:
        t = test_inputs
        Ktt = gram(self.prior.kernel, t, params["kernel"])
        Kzt = cross_covariance(self.prior.kernel, z, t, params["kernel"])
        M = solve_triangular(Lz, Kzt, lower=True)
        Fcov = Ktt - jnp.matmul(M.T, M)
    
        if not self.q.whiten:
            M = solve_triangular(Lz.T, M, lower=False)
            mz = self.prior.mean(params)(z)
            mu -= mz

        Fmu = self.prior.mean(params)(t) + jnp.matmul(M.T, mu)
        V = jnp.matmul(M.T, sqrt)
        Fcov += jnp.matmul(V, V.T)
        
        return Fmu, Fcov

    def mean(self, params: Dict) -> Callable[[Array], Array]:
        # Cholesky decomposition at inducing inputs:
        z = params["q"]["inducing_inputs"]
        nz = self.num_inducing
        Kzz = gram(self.prior.kernel, z, params["kernel"])
        Kzz += I(nz) * self.jitter
        Lz = cholesky(Kzz)

        # Variational mean:
        mu = params["q"]["mu"]
        if not self.q.whiten:
            mz = self.prior.mean(params)(z)
            mu -= mz
        
        def mean_fn(test_inputs: Array):
            t = test_inputs
            Kzt = cross_covariance(self.prior.kernel, z, t, params["kernel"])
            M = solve_triangular(Lz, Kzt, lower=True)
    
            if not self.q.whiten:
                M = solve_triangular(Lz.T, M, lower=False)

            mt = self.prior.mean(params)(t)
             
            return mt + jnp.matmul(M.T, mu)

        return mean_fn

    def variance(self, params: Dict) -> Callable[[Array], Array]:
        # Cholesky decomposition at inducing inputs:
        z = params["q"]["inducing_inputs"]
        nz = self.num_inducing
        Kzz = gram(self.prior.kernel, z, params["kernel"])
        Kzz += I(nz) * self.jitter
        Lz = cholesky(Kzz)

        # Variational sqrt cov:
        sqrt = params["q"]["sqrt"]

        def variance_fn(test_inputs: Array):
            t = test_inputs
            Ktt = gram(self.prior.kernel, t, params["kernel"])
            Kzt = cross_covariance(self.prior.kernel, z, t, params["kernel"])
            M = solve_triangular(Lz, Kzt, lower=True)
            Fcov = Ktt - jnp.matmul(M.T, M)

            if not self.q.whiten:
                M = solve_triangular(Lz.T, M, lower=False)

            V = jnp.matmul(M.T, sqrt)
            Fcov += jnp.matmul(V, V.T)

            return Fcov

        return variance_fn