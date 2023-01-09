# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc
from typing import Callable, Dict

import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap
from jaxtyping import Array, Float

from jaxlinop import identity
from jax.random import KeyArray
from jaxutils import PyTree

from .config import get_global_config
from .gps import AbstractPosterior
from .likelihoods import Gaussian
from .quadrature import gauss_hermite_quadrature
from jaxutils import Dataset
from .utils import concat_dictionaries
from .variational_families import (
    AbstractVariationalFamily,
    CollapsedVariationalGaussian,
)

import deprecation


class AbstractVariationalInference(PyTree):
    """A base class for inference and training of variational families against an extact posterior"""

    def __init__(
        self,
        posterior: AbstractPosterior,
        variational_family: AbstractVariationalFamily,
    ) -> None:
        """Initialise the variational inference module.

        Args:
            posterior (AbstractPosterior): The exact posterior distribution.
            variational_family (AbstractVariationalFamily): The variational family to be trained.
        """
        self.posterior = posterior
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood
        self.variational_family = variational_family

    def init_params(self, key: KeyArray) -> Dict:
        """Construct the parameter set used within the variational scheme adopted."""
        hyperparams = concat_dictionaries(
            {"likelihood": self.posterior.likelihood.init_params(key)},
            self.variational_family.init_params(key),
        )
        return hyperparams

    @deprecation.deprecated(
        deprecated_in="0.5.7",
        removed_in="0.6.0",
        details="Use the ``init_params`` method for parameter initialisation.",
    )
    def _initialise_params(self, key: KeyArray) -> Dict:
        """Deprecated method for initialising the GP's parameters. Succeded by ``init_params``."""
        return self.init_params(key)

    @abc.abstractmethod
    def elbo(
        self,
        train_data: Dataset,
    ) -> Callable[[Dict], Float[Array, "1"]]:
        """Placeholder method for computing the evidence lower bound function (ELBO), given a training dataset and a set of transformations that map each parameter onto the entire real line.

        Args:
            train_data (Dataset): The training dataset for which the ELBO is to be computed.

        Returns:
            Callable[[Array], Array]: A function that computes the ELBO given a set of parameters.
        """
        raise NotImplementedError


class StochasticVI(AbstractVariationalInference):
    """Stochastic Variational inference training module. The key reference is Hensman et. al., (2013) - Gaussian processes for big data."""

    def elbo(
        self, train_data: Dataset, negative: bool = False
    ) -> Callable[[Float[Array, "N D"]], Float[Array, "1"]]:
        """Compute the evidence lower bound under this model. In short, this requires evaluating the expectation of the model's log-likelihood under the variational approximation. To this, we sum the KL divergence from the variational posterior to the prior. When batching occurs, the result is scaled by the batch size relative to the full dataset size.

        Args:
            train_data (Dataset): The training data for which we should maximise the ELBO with respect to.
            negative (bool, optional): Whether or not the resultant elbo function should be negative. For gradient descent where we minimise our objective function this argument should be true as minimisation of the negative corresponds to maximisation of the ELBO. Defaults to False.

        Returns:
            Callable[[Dict, Dataset], Array]: A callable function that accepts a current parameter estimate and batch of data for which gradients should be computed.
        """

        # Constant for whether or not to negate the elbo for optimisation purposes
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def elbo_fn(params: Dict, batch: Dataset) -> Float[Array, "1"]:
            # KL[q(f(·)) || p(f(·))]
            kl = self.variational_family.prior_kl(params)

            # ∫[log(p(y|f(·))) q(f(·))] df(·)
            var_exp = self.variational_expectation(params, batch)

            # For batch size b, we compute  n/b * Σᵢ[ ∫log(p(y|f(xᵢ))) q(f(xᵢ)) df(xᵢ)] - KL[q(f(·)) || p(f(·))]
            return constant * (jnp.sum(var_exp) * train_data.n / batch.n - kl)

        return elbo_fn

    def variational_expectation(
        self, params: Dict, batch: Dataset
    ) -> Float[Array, "N 1"]:
        """Compute the expectation of our model's log-likelihood under our variational distribution. Batching can be done here to speed up computation.

        Args:
            params (Dict): The set of parameters that induce our variational approximation.
            batch (Dataset): The data batch for which the expectation should be computed for.

        Returns:
            Array: The expectation of the model's log-likelihood under our variational distribution.
        """

        # Unpack training batch
        x, y = batch.X, batch.y

        # Variational distribution q(f(·)) = N(f(·); μ(·), Σ(·, ·))
        q = self.variational_family(params)

        # Compute variational mean, μ(x), and variance, √diag(Σ(x, x)), at training inputs, x
        def q_moments(x):
            qx = q(x)
            return qx.mean(), qx.variance()

        mean, variance = vmap(q_moments)(x[:, None])

        # log(p(y|f(x)))
        link_function = self.likelihood.link_function
        log_prob = vmap(lambda f, y: link_function(params["likelihood"], f).log_prob(y))

        # ≈ ∫[log(p(y|f(x))) q(f(x))] df(x)
        expectation = gauss_hermite_quadrature(log_prob, mean, jnp.sqrt(variance), y=y)

        return expectation


class CollapsedVI(AbstractVariationalInference):
    """Collapsed variational inference for a sparse Gaussian process regression model.
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables in Sparse Gaussian Processes."""

    def __init__(
        self,
        posterior: AbstractPosterior,
        variational_family: AbstractVariationalFamily,
    ) -> None:
        """Initialise the variational inference module.

        Args:
            posterior (AbstractPosterior): The exact posterior distribution.
            variational_family (AbstractVariationalFamily): The variational family to be trained.
        """

        if not isinstance(posterior.likelihood, Gaussian):
            raise TypeError("Likelihood must be Gaussian.")

        if not isinstance(variational_family, CollapsedVariationalGaussian):
            raise TypeError("Variational family must be CollapsedVariationalGaussian.")

        super().__init__(posterior, variational_family)

    def elbo(
        self, train_data: Dataset, negative: bool = False
    ) -> Callable[[Dict], Float[Array, "1"]]:
        """Compute the evidence lower bound under this model. In short, this requires evaluating the expectation of the model's log-likelihood under the variational approximation. To this, we sum the KL divergence from the variational posterior to the prior. When batching occurs, the result is scaled by the batch size relative to the full dataset size.

        Args:
            train_data (Dataset): The training data for which we should maximise the ELBO with respect to.
            negative (bool, optional): Whether or not the resultant elbo function should be negative. For gradient descent where we minimise our objective function this argument should be true as minimisation of the negative corresponds to maximisation of the ELBO. Defaults to False.

        Returns:
            Callable[[Dict, Dataset], Array]: A callable function that accepts a current parameter estimate for which gradients should be computed.
        """

        # Unpack training data
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        m = self.variational_family.num_inducing
        jitter = get_global_config()["jitter"]

        # Constant for whether or not to negate the elbo for optimisation purposes
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def elbo_fn(params: Dict) -> Float[Array, "1"]:
            noise = params["likelihood"]["obs_noise"]
            z = params["variational_family"]["inducing_inputs"]
            Kzz = kernel.gram(params["kernel"], z)
            Kzz += identity(m) * jitter
            Kzx = kernel.cross_covariance(params["kernel"], z, x)
            Kxx_diag = vmap(kernel, in_axes=(None, 0, 0))(params["kernel"], x, x)
            μx = mean_function(params["mean_function"], x)

            Lz = Kzz.to_root()

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

            A = Lz.solve(Kzx) / jnp.sqrt(noise)

            # AAᵀ
            AAT = jnp.matmul(A, A.T)

            # B = I + AAᵀ
            B = jnp.eye(m) + AAT

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


__all__ = [
    "AbstractVariationalInference",
    "StochasticVI",
    "CollapsedVI",
]
