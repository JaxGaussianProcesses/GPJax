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
    """A variatioanl posterior object. With reference to some true posterior distribution :math:`p`, this can be used to minmise the KL-diverence between :math:`p` and a variational posterior :math:`q`."""

    posterior: AbstractPosterior
    variational_family: VariationalFamily
    jitter: Optional[float] = DEFAULT_JITTER

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """For a given set of parameters, compute the latent function's prediction under the variational approximation."""
        return self.predict(*args, **kwargs)

    @property
    def params(self) -> Dict:
        """Construct the parameter set used within the variational scheme adopted."""
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
    ) -> Callable[[Dict], Array]:
        """Placeholder method for computing the evidence lower bound function, given a training dataset and a set of transformations that map each parameter onto the entire real line.

        Args:
            train_data (Dataset): The training dataset for which the ELBO is to be computed.
            transformations (Dict): A set of functions that unconstrain each parameter.

        Returns:
            Callable[[Array], Array]: A function that computes the ELBO given a set of parameters.
        """
        raise NotImplementedError


@dataclass
class SVGP(VariationalPosterior):
    """The Sparse Variational Gaussian Process (SVGP) variational posterior. The key reference is Henman et. al., (2013) - Gaussian processes for big data."""

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood
        self.num_inducing = self.variational_family.num_inducing

    def elbo(
        self, train_data: Dataset, transformations: Dict, negative: bool = False
    ) -> Callable[[Array], Array]:
        """Compute the evidence lower bound under this model. In short, this requires evaluating the expectation of the model's log-likelihood under the variational approximation. To this, we sum the KL diverence from the variational posterior to the prior. When batching occurs, the result is scaled by the batch size relative to the full dataset size.

        Args:
            train_data (Dataset): The training data for which we should maximise the ELBO with respect to.
            transformations (Dict): The transformation set that unconstrains each parameter.
            negative (bool, optional): Whether or not the resultant elbo function should be negative. For gradient descent where we minimise our objective function this argument should be true as minimisation of the negative corresponds to maximiation of the ELBO. Defaults to False.

        Returns:
            Callable[[Dict, Dataset], Array]: A callable function that accepts a current parameter estimate and batch of data for which gradients should be computed.
        """
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def elbo_fn(params: Dict, batch: Dataset) -> Array:
            params = transform(params, transformations)
            kl = self.prior_kl(params)
            var_exp = self.variational_expectation(params, batch)

            return constant * (jnp.sum(var_exp) * train_data.n / batch.n - kl)

        return elbo_fn

    # Compute KL divergence at inducing points, KL[q(u)||p(u)]:
    def prior_kl(self, params: Dict) -> Array:
        """Compute the KL-divergence between our current variational approximation and the Gaussian process prior.

        Args:
            params (Dict): The parameters at which our variational distribution and GP prior are to be evaluated.

        Returns:
            Array: The KL-divergence between our variational approximation and the GP prior.
        """
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
        """Compute the expectation of our model's log-likelihood under our variational distribution. Batching can be done here to speed up computation.

        Args:
            params (Dict): The set of parameters that induce our variational approximation.
            batch (Dataset): The data batch for which the expectation should be computed for.

        Returns:
            Array: The expectation of the model's log-likelihood under our variational distribution.
        """
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
        """Compute the predictive mean and variance of the GP at the test inputs. A series of 1-dimensional Gaussian-Hermite quadrature schemes are used for this.

        Args:
            params (Dict): The set of parameters that are to be used to parameterise our variational approximation and GP.
            test_inputs (Array): The test inputs at which the predictive mean and variance should be computed.

        Returns:
            Tuple[Array, Array]: The predictive mean and variance of the GP at the test inputs.
        """
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
        """Compute the predictive distribution of the GP at the test inputs.

        Args:
            params (dict): The set of parameters that are to be used to parameterise our variational approximation and GP.

        Returns:
            Callable[[Array], dx.Distribution]: A function that accepts a set of test points and will return the predictive distribution at those points.
        """
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
