import abc
from typing import Callable, Dict

import jax.numpy as jnp
import jax.scipy as jsp
from chex import dataclass
from jax import vmap
from jaxtyping import f64

from .gps import AbstractPosterior
from .kernels import cross_covariance, diagonal, gram
from .likelihoods import Gaussian
from .parameters import transform
from .quadrature import gauss_hermite_quadrature
from .types import Dataset
from .utils import I, concat_dictionaries
from .variational_families import (
    AbstractVariationalFamily,
    CollapsedVariationalGaussian,
)


@dataclass
class AbstractVariationalInference:
    """A base class for inference and training of variational families against an extact posterior"""

    posterior: AbstractPosterior
    variational_family: AbstractVariationalFamily

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood

    def _initialise_params(self, key: jnp.DeviceArray) -> Dict:
        """Construct the parameter set used within the variational scheme adopted."""
        hyperparams = concat_dictionaries(
            {"likelihood": self.posterior.likelihood._initialise_params(key)},
            self.variational_family._initialise_params(key),
        )
        return hyperparams

    @abc.abstractmethod
    def elbo(
        self, train_data: Dataset, transformations: Dict
    ) -> Callable[[Dict], f64["1"]]:
        """Placeholder method for computing the evidence lower bound function (ELBO), given a training dataset and a set of transformations that map each parameter onto the entire real line.

        Args:
            train_data (Dataset): The training dataset for which the ELBO is to be computed.
            transformations (Dict): A set of functions that unconstrain each parameter.

        Returns:
            Callable[[Array], Array]: A function that computes the ELBO given a set of parameters.
        """
        raise NotImplementedError


@dataclass
class StochasticVI(AbstractVariationalInference):
    """Stochastic Variational inference training module. The key reference is Hensman et. al., (2013) - Gaussian processes for big data."""

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood
        self.num_inducing = self.variational_family.num_inducing

    def elbo(
        self, train_data: Dataset, transformations: Dict, negative: bool = False
    ) -> Callable[[f64["N D"]], f64["1"]]:
        """Compute the evidence lower bound under this model. In short, this requires evaluating the expectation of the model's log-likelihood under the variational approximation. To this, we sum the KL divergence from the variational posterior to the prior. When batching occurs, the result is scaled by the batch size relative to the full dataset size.

        Args:
            train_data (Dataset): The training data for which we should maximise the ELBO with respect to.
            transformations (Dict): The transformation set that unconstrains each parameter.
            negative (bool, optional): Whether or not the resultant elbo function should be negative. For gradient descent where we minimise our objective function this argument should be true as minimisation of the negative corresponds to maximisation of the ELBO. Defaults to False.

        Returns:
            Callable[[Dict, Dataset], Array]: A callable function that accepts a current parameter estimate and batch of data for which gradients should be computed.
        """
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def elbo_fn(params: Dict, batch: Dataset) -> f64["1"]:
            params = transform(params, transformations)

            # KL[q(f(·)) || p(f(·))]
            kl = self.variational_family.prior_kl(params)

            # ∫[log(p(y|f(x))) q(f(x))] df(x)
            var_exp = self.variational_expectation(params, batch)

            # For batch size b, we compute  n/b * Σᵢ[ ∫log(p(y|f(xᵢ))) q(f(xᵢ)) df(xᵢ)] - KL[q(f(·)) || p(f(·))]
            return constant * (jnp.sum(var_exp) * train_data.n / batch.n - kl)

        return elbo_fn

    def variational_expectation(self, params: Dict, batch: Dataset) -> f64["N 1"]:
        """Compute the expectation of our model's log-likelihood under our variational distribution. Batching can be done here to speed up computation.

        Args:
            params (Dict): The set of parameters that induce our variational approximation.
            batch (Dataset): The data batch for which the expectation should be computed for.

        Returns:
            Array: The expectation of the model's log-likelihood under our variational distribution.
        """
        x, y = batch.X, batch.y

        # q(f(x))
        predictive_dist = vmap(self.variational_family.predict(params))(x[:, None])
        mean = predictive_dist.mean().val.reshape(-1, 1)
        variance = predictive_dist.variance().val.reshape(-1, 1)

        # log(p(y|f(x)))
        log_prob = vmap(
            lambda f, y: self.likelihood.link_function(
                f, params["likelihood"]
            ).log_prob(y)
        )

        # ≈ ∫[log(p(y|f(x))) q(f(x))] df(x)
        expectation = gauss_hermite_quadrature(log_prob, mean, variance, y=y)

        return expectation


@dataclass
class CollapsedVI(AbstractVariationalInference):
    """Collapsed variational inference for a sparse Gaussian process regression model.
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables in Sparse Gaussian Processes."""

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood
        self.num_inducing = self.variational_family.num_inducing

        if not isinstance(self.likelihood, Gaussian):
            raise TypeError("Likelihood must be Gaussian.")

        if not isinstance(self.variational_family, CollapsedVariationalGaussian):
            raise TypeError("Variational family must be CollapsedVariationalGaussian.")

    def elbo(
        self, train_data: Dataset, transformations: Dict, negative: bool = False
    ) -> Callable[[dict], f64["1"]]:
        """Compute the evidence lower bound under this model. In short, this requires evaluating the expectation of the model's log-likelihood under the variational approximation. To this, we sum the KL divergence from the variational posterior to the prior. When batching occurs, the result is scaled by the batch size relative to the full dataset size.

        Args:
            train_data (Dataset): The training data for which we should maximise the ELBO with respect to.
            transformations (Dict): The transformation set that unconstrains each parameter.
            negative (bool, optional): Whether or not the resultant elbo function should be negative. For gradient descent where we minimise our objective function this argument should be true as minimisation of the negative corresponds to maximisation of the ELBO. Defaults to False.

        Returns:
            Callable[[Dict, Dataset], Array]: A callable function that accepts a current parameter estimate for which gradients should be computed.
        """
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        x, y, n = train_data.X, train_data.y, train_data.n

        m = self.num_inducing

        def elbo_fn(params: Dict) -> f64["1"]:
            params = transform(params, transformations)
            noise = params["likelihood"]["obs_noise"]
            z = params["variational_family"]["inducing_inputs"]
            Kzz = gram(self.prior.kernel, z, params["kernel"])
            Kzz += I(m) * self.variational_family.jitter
            Kzx = cross_covariance(self.prior.kernel, z, x, params["kernel"])
            Kxx_diag = vmap(self.prior.kernel, in_axes=(0, 0, None))(
                x, x, params["kernel"]
            )
            μx = self.prior.mean_function(x, params["mean_function"])

            Lz = jnp.linalg.cholesky(Kzz)

            # Notation and derivation:
            #
            # Let Q = KxzKzz⁻¹Kzx, we must compute the log normal pdf:
            #
            #   log N(y; μx, σ²I + Q) = -nπ - n/2 log|σ²I + Q| - 1/2 (y - μx)ᵀ (σ²I + Q)⁻¹ (y - μx).
            #
            # The log determinant |σ²I + Q| is computed via applying the matrix determinant lemma
            #
            #   |σ²I + Q| = log|σ²I| + log|I + Lz⁻¹ Kzx (σ²I)⁻¹ Kxz Lz⁻¹| = log(σ²) +  log|B|,
            #
            #   with B = I + AAᵀ and A = Lz⁻¹ Kzx / σ.
            #
            # Similary we apply matrix inversion lemma to invert σ²I + Q
            #
            #   (σ²I + Q)⁻¹ = (Iσ²)⁻¹ - (Iσ²)⁻¹ Kxz Lz⁻ᵀ (I + Lz⁻¹ Kzx (Iσ²)⁻¹ Kxz Lz⁻ᵀ )⁻¹ Lz⁻¹ Kzx (Iσ²)⁻¹
            #               = (Iσ²)⁻¹ - (Iσ²)⁻¹ σAᵀ (I + σA (Iσ²)⁻¹ σAᵀ)⁻¹ σA (Iσ²)⁻¹
            #               = I/σ² - Aᵀ B⁻¹ A/σ²,
            #
            # giving the quadratic term as
            #
            #   (y - μx)ᵀ (σ²I + Q)⁻¹ (y - μx) = [(y - μx)ᵀ(y - µx)  - (y - μx)ᵀ Aᵀ B⁻¹ A (y - μx)]/σ²,
            #
            #   with A and B defined as above.

            A = jsp.linalg.solve_triangular(Lz, Kzx, lower=True) / jnp.sqrt(noise)

            # AAᵀ
            AAT = jnp.matmul(A, A.T)

            # B = I + AAᵀ
            B = I(m) + AAT

            # LLᵀ = I + AAᵀ
            L = jnp.linalg.cholesky(B)

            # log|B| = 2 trace(log|L|) = 2 Σᵢ log Lᵢᵢ  [since |B| = |LLᵀ| = |L|²  => log|B| = 2 log|L|, and |L| = Πᵢ Lᵢᵢ]
            log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))

            diff = y - μx

            # L⁻¹ A (y - μx)
            L_inv_A_diff = jsp.linalg.solve_triangular(
                L, jnp.matmul(A, diff), lower=True
            )

            # (y - μx)ᵀ (Iσ² + Q)⁻¹ (y - μx)
            quad = (jnp.sum(diff**2) - jnp.sum(L_inv_A_diff**2)) / noise

            # 2 * log N(y; μx, Iσ² + Q)
            two_log_prob = -n * jnp.log(2.0 * jnp.pi * noise) - log_det_B - quad

            # 1/σ² tr(Kxx - Q) [Trace law tr(AB) = tr(BA) => tr(KxzKzz⁻¹Kzx) = tr(KxzLz⁻ᵀLz⁻¹Kzx) = tr(Lz⁻¹Kzx KxzLz⁻ᵀ) = trace(σ²AAᵀ)]
            two_trace = jnp.sum(Kxx_diag) / noise - jnp.trace(AAT)

            # log N(y; μx, Iσ² + KxzKzz⁻¹Kzx) - 1/2σ² tr(Kxx - KxzKzz⁻¹Kzx)
            return constant * (two_log_prob - two_trace).squeeze() / 2.0

        return elbo_fn
