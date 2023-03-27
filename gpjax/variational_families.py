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
from dataclasses import dataclass
from typing import Any

import deprecation
import distrax as dx
import jax.numpy as jnp
import jax.scipy as jsp
import tensorflow_probability.substrates.jax.bijectors as tfb
from jaxtyping import Array, Float
from jaxutils import Dataset, PyTree

from .config import get_global_config
from .gaussian_distribution import GaussianDistribution
from .gps import Prior
from .likelihoods import AbstractLikelihood, Gaussian
from .linops import DenseLinearOperator, LowerTriangularLinearOperator, identity
from .utils import concat_dictionaries


@dataclass
class AbstractVariationalFamily(Module):
    """
    Abstract base class used to represent families of distributions that can be
    used within variational inference.
    """

    posterior: AbstractPosterior

    def __call__(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        """For a given set of parameters, compute the latent function's prediction
        under the variational approximation.

        Args:
            *args (Any): Arguments of the variational family's `predict` method.
            **kwargs (Any): Keyword arguments of the variational family's `predict`
                method.

        Returns:
            GaussianDistribution: The output of the variational family's `predict` method.
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        """Predict the GP's output given the input.

        Args:
            *args (Any): Arguments of the variational family's ``predict``
            method.
            **kwargs (Any): Keyword arguments of the variational family's
            ``predict`` method.

        Returns:
            GaussianDistribution: The output of the variational family's ``predict`` method.
        """
        raise NotImplementedError


@dataclass
class AbstractVariationalGaussian(AbstractVariationalFamily):
    """The variational Gaussian family of probability distributions."""

    inducing_inputs: Float[Array, "N D"]
    jitter: Float[Array, "1"] = static_field(1e-6)

    @property
    def num_inducing(self) -> int:
        """The number of inducing inputs."""
        return self.inducing_inputs.shape[0]


@dataclass
class VariationalGaussian(AbstractVariationalGaussian):
    """The variational Gaussian family of probability distributions.

    The variational family is q(f(·)) = ∫ p(f(·)|u) q(u) du, where
    :math:`u = f(z)` are the function values at the inducing inputs z
    and the distribution over the inducing inputs is
    :math:`q(u) = \\mathcal{N}(\\mu, S)`.  We parameterise this over
    :math:`\\mu` and sqrt with S = sqrt sqrtᵀ.
    """

    variational_mean: Float[Array, "N 1"] = param_field(None)
    variational_root_covariance: Float[Array, "N N"] = param_field(
        None, bijector=tfb.FillTriangular()
    )

    def __post_init__(self) -> None:
        if self.variational_mean is None:
            self.variational_mean = jnp.zeros((self.num_inducing, 1))

        if self.variational_root_covariance is None:
            self.variational_root_covariance = jnp.eye(self.num_inducing)

    def prior_kl(self) -> Float[Array, "1"]:
        """
        Compute the KL-divergence between our variational approximation and the
        Gaussian process prior.

        For this variational family, we have KL[q(f(·))||p(·)] = KL[q(u)||p(u)]
        = KL[ N(μ, S) || N(μz, Kzz) ], where u = f(z) and z are the inducing
        inputs.

        Returns:
             Float[Array, "1"]: The KL-divergence between our variational
                approximation and the GP prior.
        """

        # Unpack variational parameters
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        z = self.inducing_inputs
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        μz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz += identity(m) * self.jitter

        sqrt = LowerTriangularLinearOperator.from_dense(sqrt)
        S = DenseLinearOperator.from_root(sqrt)

        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.atleast_1d(μz.squeeze()), scale=Kzz)

        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        """
        Compute the predictive distribution of the GP at the test inputs t.

        This is the integral q(f(t)) = ∫ p(f(t)|u) q(u) du, which can be
        computed in closed form as:

            N[f(t); μt + Ktz Kzz⁻¹ (μ - μz),  Ktt - Ktz Kzz⁻¹ Kzt + Ktz Kzz⁻¹ S Kzz⁻¹ Kzt ].

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to make a prediction.

        Returns:
            GaussianDistribution: The predictive distribution of the low-rank GP at the test inputs.
        """

        # Unpack variational parameters
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        z = self.inducing_inputs
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzz = kernel.gram(z)
        Kzz += identity(m) * self.jitter
        Lz = Kzz.to_root()
        μz = mean_function(z)

        # Unpack test inputs
        t, n_test = test_inputs, test_inputs.shape[0]

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        μt = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = Lz.solve(Kzt)

        # Kzz⁻¹ Kzt
        Kzz_inv_Kzt = Lz.T.solve(Lz_inv_Kzt)

        # Ktz Kzz⁻¹ sqrt
        Ktz_Kzz_inv_sqrt = jnp.matmul(Kzz_inv_Kzt.T, sqrt)

        # μt + Ktz Kzz⁻¹ (μ - μz)
        mean = μt + jnp.matmul(Kzz_inv_Kzt.T, mu - μz)

        # Ktt - Ktz Kzz⁻¹ Kzt  +  Ktz Kzz⁻¹ S Kzz⁻¹ Kzt  [recall S = sqrt sqrtᵀ]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Kzz_inv_sqrt, Ktz_Kzz_inv_sqrt.T)
        )
        covariance += identity(n_test) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


@dataclass
class WhitenedVariationalGaussian(VariationalGaussian):
    """
    The whitened variational Gaussian family of probability distributions.

    The variational family is q(f(·)) = ∫ p(f(·)|u) q(u) du, where u = f(z)
    are the function values at the inducing inputs z and the distribution over
    the inducing inputs is q(u) = N(Lz μ + mz, Lz S Lzᵀ). We parameterise this
    over μ and sqrt with S = sqrt sqrtᵀ.
    """

    def prior_kl(self) -> Float[Array, "1"]:
        """Compute the KL-divergence between our variational approximation and
        the Gaussian process prior.

        For this variational family, we have KL[q(f(·))||p(·)] = KL[q(u)||p(u)] = KL[N(μ, S)||N(0, I)].

        Returns:
            Float[Array, "1"]: The KL-divergence between our variational
                approximation and the GP prior.
        """

        # Unpack variational parameters
        mu = self.variational_mean
        sqrt = self.variational_root_covariance

        sqrt = LowerTriangularLinearOperator.from_dense(sqrt)
        S = DenseLinearOperator.from_root(sqrt)

        # Compute whitened KL divergence
        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.zeros_like(jnp.atleast_1d(mu.squeeze())))
        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        """Compute the predictive distribution of the GP at the test inputs t.

        This is the integral q(f(t)) = ∫ p(f(t)|u) q(u) du, which can be computed in closed form as

            N[f(t); μt  +  Ktz Lz⁻ᵀ μ,  Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Lz⁻ᵀ S Lz⁻¹ Kzt].

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to make a prediction.

        Returns:
            GaussianDistribution: The predictive distribution of the low-rank GP at the test inputs.
        """

        # Unpack variational parameters
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        z = self.inducing_inputs
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzz = kernel.gram(z)
        Kzz += identity(m) * self.jitter
        Lz = Kzz.to_root()

        # Unpack test inputs
        t, n_test = test_inputs, test_inputs.shape[0]

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        μt = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = Lz.solve(Kzt)

        # Ktz Lz⁻ᵀ sqrt
        Ktz_Lz_invT_sqrt = jnp.matmul(Lz_inv_Kzt.T, sqrt)

        # μt  +  Ktz Lz⁻ᵀ μ
        mean = μt + jnp.matmul(Lz_inv_Kzt.T, mu)

        # Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Lz⁻ᵀ S Lz⁻¹ Kzt  [recall S = sqrt sqrtᵀ]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Lz_invT_sqrt, Ktz_Lz_invT_sqrt.T)
        )
        covariance += identity(n_test) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


@dataclass
class NaturalVariationalGaussian(AbstractVariationalGaussian):
    """The natural variational Gaussian family of probability distributions.

    The variational family is q(f(·)) = ∫ p(f(·)|u) q(u) du, where u = f(z) are the function values at the inducing inputs z
    and the distribution over the inducing inputs is q(u) = N(μ, S). Expressing the variational distribution, in the form of the
    exponential family, q(u) = exp(θᵀ T(u) - a(θ)), gives rise to the natural parameterisation θ = (θ₁, θ₂) = (S⁻¹μ, -S⁻¹/2), to perform
    model inference, where T(u) = [u, uuᵀ] are the sufficient statistics.
    """

    natural_vector: Float[Array, "M 1"] = None
    natural_matrix: Float[Array, "M M"] = None

    def __post_init__(self):
        if self.natural_vector is None:
            self.natural_vector = jnp.zeros((self.num_inducing, 1))

        if self.natural_matrix is None:
            self.natural_matrix = -0.5 * jnp.eye(self.num_inducing)

    def prior_kl(self) -> Float[Array, "1"]:
        """Compute the KL-divergence between our current variational approximation and the Gaussian process prior.

        For this variational family, we have KL[q(f(·))||p(·)] = KL[q(u)||p(u)] = KL[N(μ, S)||N(mz, Kzz)],

        with μ and S computed from the natural parameterisation θ = (S⁻¹μ, -S⁻¹/2).

        Returns:
            Float[Array, "1"]: The KL-divergence between our variational approximation and the GP prior.
        """

        # Unpack variational parameters
        natural_vector = self.natural_vector
        natural_matrix = self.natural_matrix
        z = self.inducing_inputs
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # S⁻¹ = -2θ₂
        S_inv = -2 * natural_matrix
        S_inv += jnp.eye(m) * self.jitter

        # Compute L⁻¹, where LLᵀ = S, via a trick found in the NumPyro source code and https://nbviewer.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril:
        sqrt_inv = jnp.swapaxes(
            jnp.linalg.cholesky(S_inv[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
        )

        # L = (L⁻¹)⁻¹I
        sqrt = jsp.linalg.solve_triangular(sqrt_inv, jnp.eye(m), lower=True)
        sqrt = LowerTriangularLinearOperator.from_dense(sqrt)

        # S = LLᵀ:
        S = DenseLinearOperator.from_root(sqrt)

        # μ = Sθ₁
        mu = S @ natural_vector

        μz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz += identity(m) * self.jitter

        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.atleast_1d(μz.squeeze()), scale=Kzz)

        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        """Compute the predictive distribution of the GP at the test inputs t.

        This is the integral q(f(t)) = ∫ p(f(t)|u) q(u) du, which can be computed in closed form as

             N[f(t); μt + Ktz Kzz⁻¹ (μ - μz),  Ktt - Ktz Kzz⁻¹ Kzt + Ktz Kzz⁻¹ S Kzz⁻¹ Kzt ],

        with μ and S computed from the natural parameterisation θ = (S⁻¹μ, -S⁻¹/2).

        Returns:
            GaussianDistribution: A function that accepts a set of test points and will return the predictive distribution at those points.
        """

        # Unpack variational parameters
        natural_vector = self.natural_vector
        natural_matrix = self.natural_matrix
        z = self.inducing_inputs
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # S⁻¹ = -2θ₂
        S_inv = -2 * natural_matrix
        S_inv += jnp.eye(m) * self.jitter

        # Compute L⁻¹, where LLᵀ = S, via a trick found in the NumPyro source code and https://nbviewer.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril:
        sqrt_inv = jnp.swapaxes(
            jnp.linalg.cholesky(S_inv[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
        )

        # L = (L⁻¹)⁻¹I
        sqrt = jsp.linalg.solve_triangular(sqrt_inv, jnp.eye(m), lower=True)

        # S = LLᵀ:
        S = jnp.matmul(sqrt, sqrt.T)

        # μ = Sθ₁
        mu = jnp.matmul(S, natural_vector)

        Kzz = kernel.gram(z)
        Kzz += identity(m) * self.jitter
        Lz = Kzz.to_root()
        μz = mean_function(z)

        # Unpack test inputs
        t, n_test = test_inputs, test_inputs.shape[0]

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        μt = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = Lz.solve(Kzt)

        # Kzz⁻¹ Kzt
        Kzz_inv_Kzt = Lz.T.solve(Lz_inv_Kzt)

        # Ktz Kzz⁻¹ L
        Ktz_Kzz_inv_L = jnp.matmul(Kzz_inv_Kzt.T, sqrt)

        # μt  +  Ktz Kzz⁻¹ (μ  -  μz)
        mean = μt + jnp.matmul(Kzz_inv_Kzt.T, mu - μz)

        # Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Kzz⁻¹ S Kzz⁻¹ Kzt  [recall S = LLᵀ]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Kzz_inv_L, Ktz_Kzz_inv_L.T)
        )
        covariance += identity(n_test) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


@dataclass
class ExpectationVariationalGaussian(AbstractVariationalGaussian):
    """The natural variational Gaussian family of probability distributions.

    The variational family is q(f(·)) = ∫ p(f(·)|u) q(u) du, where u = f(z) are the function values at the inducing inputs z
    and the distribution over the inducing inputs is q(u) = N(μ, S). Expressing the variational distribution, in the form of the
    exponential family, q(u) = exp(θᵀ T(u) - a(θ)), gives rise to the natural parameterisation θ = (θ₁, θ₂) = (S⁻¹μ, -S⁻¹/2) and
    sufficient statistics T(u) = [u, uuᵀ]. The expectation parameters are given by η = ∫ T(u) q(u) du. This gives a parameterisation,
    η = (η₁, η₁) = (μ, S + uuᵀ) to perform model inference over.
    """

    expectation_vector: Float[Array, "M 1"] = None
    expectation_matrix: Float[Array, "M M"] = None

    def __post_init__(self):
        if self.expectation_vector is None:
            self.expectation_vector = jnp.zeros((self.num_inducing, 1))
        if self.expectation_matrix is None:
            self.expectation_matrix = jnp.eye(self.num_inducing)

    def prior_kl(self) -> Float[Array, "1"]:
        """Compute the KL-divergence between our current variational approximation and the Gaussian process prior.

        For this variational family, we have KL[q(f(·))||p(·)] = KL[q(u)||p(u)] = KL[N(μ, S)||N(mz, Kzz)],

        with μ and S computed from the expectation parameterisation η = (μ, S + uuᵀ).

        Args:
            params (Dict): The parameters at which our variational distribution and GP prior are to be evaluated.

        Returns:
            Float[Array, "1"]: The KL-divergence between our variational approximation and the GP prior.
        """

        # Unpack variational parameters
        expectation_vector = self.expectation_vector
        expectation_matrix = self.expectation_matrix
        z = self.inducing_inputs
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # μ = η₁
        mu = expectation_vector

        # S = η₂ - η₁ η₁ᵀ
        S = expectation_matrix - jnp.outer(mu, mu)
        S = DenseLinearOperator(S)
        S += identity(m) * self.jitter

        μz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz += identity(m) * self.jitter

        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.atleast_1d(μz.squeeze()), scale=Kzz)

        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        """Compute the predictive distribution of the GP at the test inputs t.

        This is the integral q(f(t)) = ∫ p(f(t)|u) q(u) du, which can be computed in closed form as

             N[f(t); μt + Ktz Kzz⁻¹ (μ - μz),  Ktt - Ktz Kzz⁻¹ Kzt + Ktz Kzz⁻¹ S Kzz⁻¹ Kzt ],

        with μ and S computed from the expectation parameterisation η = (μ, S + uuᵀ).

        Returns:
            GaussianDistribution: The predictive distribution of the GP at the test inputs t.
        """

        # Unpack variational parameters
        expectation_vector = self.expectation_vector
        expectation_matrix = self.expectation_matrix
        z = self.inducing_inputs
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # μ = η₁
        mu = expectation_vector

        # S = η₂ - η₁ η₁ᵀ
        S = expectation_matrix - jnp.matmul(mu, mu.T)
        S = DenseLinearOperator(S)
        S += identity(m) * self.jitter

        # S = sqrt sqrtᵀ
        sqrt = S.to_root().to_dense()

        Kzz = kernel.gram(z)
        Kzz += identity(m) * self.jitter
        Lz = Kzz.to_root()
        μz = mean_function(z)

        # Unpack test inputs
        t, n_test = test_inputs, test_inputs.shape[0]

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        μt = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = Lz.solve(Kzt)

        # Kzz⁻¹ Kzt
        Kzz_inv_Kzt = Lz.T.solve(Lz_inv_Kzt)

        # Ktz Kzz⁻¹ sqrt
        Ktz_Kzz_inv_sqrt = jnp.matmul(Kzz_inv_Kzt.T, sqrt)

        # μt  +  Ktz Kzz⁻¹ (μ  -  μz)
        mean = μt + jnp.matmul(Kzz_inv_Kzt.T, mu - μz)

        # Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Kzz⁻¹ S Kzz⁻¹ Kzt  [recall S = sqrt sqrtᵀ]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Kzz_inv_sqrt, Ktz_Kzz_inv_sqrt.T)
        )
        covariance += identity(n_test) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


@dataclass
class CollapsedVariationalGaussian(AbstractVariationalGaussian):
    """Collapsed variational Gaussian family of probability distributions.
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables in Sparse Gaussian Processes.
    """

    def __post_init__(self):
        if not isinstance(self.posterior.likelihood, Gaussian):
            raise TypeError("Likelihood must be Gaussian.")

    def predict(
        self, test_inputs: Float[Array, "N D"], train_data: Dataset
    ) -> GaussianDistribution:
        """Compute the predictive distribution of the GP at the test inputs.

        Args:
            train_data (Dataset): The training data that was used to fit the GP.

        Returns:
            GaussianDistribution: The predictive distribution of the collapsed variational Gaussian process at the test inputs t.
        """

        # Unpack test inputs
        t, n_test = test_inputs, test_inputs.shape[0]

        # Unpack training data
        x, y = train_data.X, train_data.y

        # Unpack variational parameters
        noise = self.posterior.likelihood.obs_noise
        z = self.inducing_inputs
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzx = kernel.cross_covariance(z, x)
        Kzz = kernel.gram(z)
        Kzz += identity(m) * self.jitter

        # Lz Lzᵀ = Kzz
        Lz = Kzz.to_root()

        # Lz⁻¹ Kzx
        Lz_inv_Kzx = Lz.solve(Kzx)

        # A = Lz⁻¹ Kzt / σ
        A = Lz_inv_Kzx / jnp.sqrt(noise)

        # AAᵀ
        AAT = jnp.matmul(A, A.T)

        # LLᵀ = I + AAᵀ
        L = jnp.linalg.cholesky(jnp.eye(m) + AAT)

        μx = mean_function(x)
        diff = y - μx

        # Lz⁻¹ Kzx (y - μx)
        Lz_inv_Kzx_diff = jsp.linalg.cho_solve((L, True), jnp.matmul(Lz_inv_Kzx, diff))

        # Kzz⁻¹ Kzx (y - μx)
        Kzz_inv_Kzx_diff = Lz.T.solve(Lz_inv_Kzx_diff)

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        μt = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = Lz.solve(Kzt)

        # L⁻¹ Lz⁻¹ Kzt
        L_inv_Lz_inv_Kzt = jsp.linalg.solve_triangular(L, Lz_inv_Kzt, lower=True)

        # μt + 1/σ² Ktz Kzz⁻¹ Kzx (y - μx)
        mean = μt + jnp.matmul(Kzt.T / noise, Kzz_inv_Kzx_diff)

        # Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Lz⁻¹ (I + AAᵀ)⁻¹ Lz⁻¹ Kzt
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(L_inv_Lz_inv_Kzt.T, L_inv_Lz_inv_Kzt)
        )
        covariance += identity(n_test) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


__all__ = [
    "AbstractVariationalFamily",
    "AbstractVariationalGaussian",
    "VariationalGaussian",
    "WhitenedVariationalGaussian",
    "NaturalVariationalGaussian",
    "ExpectationVariationalGaussian",
    "CollapsedVariationalGaussian",
]
