from typing import Callable, Dict

import distrax as dx
import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular


from gpjax.kernels import cross_covariance, gram
from gpjax.parameters import transform
from gpjax.types import Array, Dataset
from gpjax.utils import I

from .quadrature import gauss_hermite_quadrature
from .variational import VariationalPosterior

@dataclass
class SVGP(VariationalPosterior):

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood
        self.num_inducing = self.variational_family.num_inducing

    def elbo(self, train_data: Dataset, transformations: Dict) -> Callable[[Array], Array]:
        def elbo_fn(params: Dict, batch: Dataset) -> Array:
            params = transform(params, transformations)
            kl = self.prior_kl(params)
            var_exp = self.variational_expectation(params, batch)

            return jnp.sum(var_exp) * train_data.n / batch.n - kl

        return elbo_fn

    # Compute KL divergence at inducing points, KL[q(u)||p(u)]:
    def prior_kl(self, params: Dict) -> Array:
        mu = params["variational_family"]["variational_mean"]
        sqrt = params["variational_family"]["variational_root_covariance"]
        nz = self.num_inducing

        qu = dx.MultivariateNormalTri(mu.squeeze(), sqrt)

        if not self.variational_family.whiten:
            z = params["variational_family"]["inducing_inputs"]
            mz = self.prior.mean_function(z, params["mean_function"])
            Kzz = gram(self.prior.kernel, z, params["kernel"])
            Kzz += I(nz) * self.jitter
            Lz = cholesky(Kzz)
            pu = dx.MultivariateNormalTri(mz.squeeze(), Lz)

        else:
            pu = dx.MultivariateNormalDiag(jnp.zeros(nz))

        return qu.kl_divergence(pu)

    # Compute expectation, ∫ log(p(y|F)) q(F) dF, through Gauss-Hermite quadrature:
    def variational_expectation(self, params: Dict, batch: Dataset) -> Array:
        x, y = batch.X, batch.y

        Fmu, Fvar = vmap(self.pred_moments, in_axes=(None, 0))(params, x[:, jnp.newaxis, :])

        # Get log(p(y|F)) function for current likelihood parameter values:
        def log_prob(F, y):
            return self.likelihood.link_function(F, params["likelihood"]).log_prob(y)

        return gauss_hermite_quadrature(log_prob, Fmu.squeeze(1), Fvar.squeeze(1), y=y)

    # Computes predictive moments for Gauss-Hermite quadrature:
    def pred_moments(self, params: Dict, test_inputs: Array) -> Array:
        mu = params["variational_family"]["variational_mean"]
        sqrt = params["variational_family"]["variational_root_covariance"]

        # Cholesky decomposition at inducing inputs:
        z = params["variational_family"]["inducing_inputs"]
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

        if not self.variational_family.whiten:
            M = solve_triangular(Lz.T, M, lower=False)
            mz = self.prior.mean(params)(z)
            mu -= mz

        Fmu = self.prior.mean(params)(t) + jnp.matmul(M.T, mu)
        V = jnp.matmul(M.T, sqrt)
        Fcov += jnp.matmul(V, V.T)

        return Fmu, Fcov

    def mean(self, params: Dict) -> Callable[[Array], Array]:
        # Cholesky decomposition at inducing inputs:
        z = params["variational_family"]["inducing_inputs"]
        nz = self.num_inducing
        Kzz = gram(self.prior.kernel, z, params["kernel"])
        Kzz += I(nz) * self.jitter
        Lz = cholesky(Kzz)

        # Variational mean:
        mu = params["variational_family"]["variational_mean"]
        if not self.variational_family.whiten:
            mz = self.prior.mean(params)(z)
            mu -= mz

        def mean_fn(test_inputs: Array):
            t = test_inputs
            Kzt = cross_covariance(self.prior.kernel, z, t, params["kernel"])
            M = solve_triangular(Lz, Kzt, lower=True)

            if not self.variational_family.whiten:
                M = solve_triangular(Lz.T, M, lower=False)

            mt = self.prior.mean(params)(t)

            return mt + jnp.matmul(M.T, mu)

        return mean_fn

    def variance(self, params: Dict) -> Callable[[Array], Array]:
        # Cholesky decomposition at inducing inputs:
        z = params["variational_family"]["inducing_inputs"]
        nz = self.num_inducing
        Kzz = gram(self.prior.kernel, z, params["kernel"])
        Kzz += I(nz) * self.jitter
        Lz = cholesky(Kzz)

        # Variational sqrt cov:
        sqrt = params["variational_family"]["variational_root_covariance"]

        def variance_fn(test_inputs: Array):
            t = test_inputs
            Ktt = gram(self.prior.kernel, t, params["kernel"])
            Kzt = cross_covariance(self.prior.kernel, z, t, params["kernel"])
            M = solve_triangular(Lz, Kzt, lower=True)
            Fcov = Ktt - jnp.matmul(M.T, M)

            if not self.variational_family.whiten:
                M = solve_triangular(Lz.T, M, lower=False)

            V = jnp.matmul(M.T, sqrt)
            Fcov += jnp.matmul(V, V.T)

            return Fcov

        return variance_fn
