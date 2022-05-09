import abc
from typing import Any, Callable, Dict, Optional, Tuple

import distrax as dx
import jax.numpy as jnp
from chex import dataclass
from jax import vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular

from .config import get_defaults
from .gps import AbstractPosterior
from .kernels import cross_covariance, gram
from .parameters import transform
from .quadrature import gauss_hermite_quadrature
from .types import Array, Dataset
from .utils import I, concat_dictionaries
from .variational import VariationalFamily

DEFAULT_JITTER = get_defaults()["jitter"]


@dataclass
class VariationalPosterior:
    posterior: AbstractPosterior
    variational_family: VariationalFamily
    jitter: Optional[float] = DEFAULT_JITTER

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.predict(*args, **kwargs)

    @property
    def params(self) -> Dict:
        hyperparams = concat_dictionaries(
            self.posterior.params,
            {"variational_family": self.variational_family.params},
        )
        return hyperparams

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> dx.Distribution:
        """Predict the GP's output given the input."""
        raise NotImplementedError

    @abc.abstractmethod
    def elbo(
        self, train_data: Dataset, transformations: Dict
    ) -> Callable[[Array], Array]:
        raise NotImplementedError


@dataclass
class SVGP(VariationalPosterior):
    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood
        self.num_inducing = self.variational_family.num_inducing

    def elbo(
        self, train_data: Dataset, transformations: Dict, negative: bool = False
    ) -> Callable[[Array], Array]:
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def elbo_fn(params: Dict, batch: Dataset) -> Array:
            params = transform(params, transformations)
            kl = self.prior_kl(params)
            var_exp = self.variational_expectation(params, batch)

            return constant * (jnp.sum(var_exp) * train_data.n / batch.n - kl)

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

    def variational_expectation(self, params: Dict, batch: Dataset) -> Array:
        x, y = batch.X, batch.y

        Fmu, Fvar = vmap(self.pred_moments, in_axes=(None, 0))(
            params, x[:, jnp.newaxis, :]
        )

        # Get log(p(y|F)) function for current likelihood parameter values:
        def log_prob(F, y):
            return self.likelihood.link_function(F, params["likelihood"]).log_prob(y)

        return gauss_hermite_quadrature(log_prob, Fmu.squeeze(1), Fvar.squeeze(1), y=y)

    # Computes predictive moments for Gauss-Hermite quadrature:
    def pred_moments(self, params: Dict, test_inputs: Array) -> Tuple[Array, Array]:
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
            mz = self.prior.mean(params)(z).reshape(-1, 1)
            mu -= mz

        Fmu = self.prior.mean(params)(t).reshape(-1, 1) + jnp.matmul(M.T, mu)
        V = jnp.matmul(M.T, sqrt)
        Fcov += jnp.matmul(V, V.T)

        return Fmu, Fcov

    def predict(self, params: dict) -> Callable[[Array], dx.Distribution]:
        # Cholesky decomposition at inducing inputs:
        z = params["variational_family"]["inducing_inputs"]
        nz = self.num_inducing
        Kzz = gram(self.prior.kernel, z, params["kernel"])
        Kzz += I(nz) * self.jitter
        Lz = cholesky(Kzz)

        # Variational mean:
        mu = params["variational_family"]["variational_mean"]
        if not self.variational_family.whiten:
            mz = self.prior.mean(params)(z).reshape(-1, 1)
            mu -= mz

        # Variational sqrt cov:
        sqrt = params["variational_family"]["variational_root_covariance"]

        def predict_fn(test_inputs: Array) -> dx.Distribution:
            t = test_inputs
            Ktt = gram(self.prior.kernel, t, params["kernel"])
            Kzt = cross_covariance(self.prior.kernel, z, t, params["kernel"])
            M = solve_triangular(Lz, Kzt, lower=True)
            Fcov = Ktt - jnp.matmul(M.T, M)

            if not self.variational_family.whiten:
                M = solve_triangular(Lz.T, M, lower=False)

            mt = self.prior.mean(params)(t).reshape(-1, 1)
            mean = mt + jnp.matmul(M.T, mu)

            V = jnp.matmul(M.T, sqrt)
            covariance = Fcov + jnp.matmul(V, V.T)

            return dx.MultivariateNormalFullCovariance(
                jnp.atleast_1d(mean.squeeze()), covariance
            )

        return predict_fn
