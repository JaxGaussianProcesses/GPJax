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

from beartype.typing import (
    Any,
    Tuple,
)
import cola
from jax import (
    value_and_grad,
    vmap,
)
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from gpjax.dataset import Dataset
from gpjax.gaussian_distribution import GaussianDistribution
from gpjax.gps import AbstractPosterior
from gpjax.likelihoods import Gaussian
from gpjax.lower_cholesky import lower_cholesky
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class AbstractVariationalFamily(Module):
    r"""
    Abstract base class used to represent families of distributions that can be
    used within variational inference.
    """

    posterior: AbstractPosterior

    def __call__(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        r"""Evaluate the variational family's density.

        For a given set of parameters, compute the latent function's prediction
        under the variational approximation.

        Args:
            *args (Any): Arguments of the variational family's `predict` method.
            **kwargs (Any): Keyword arguments of the variational family's `predict`
                method.

        Returns
        -------
            GaussianDistribution: The output of the variational family's `predict` method.
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        r"""Predict the GP's output given the input.

        Args:
            *args (Any): Arguments of the variational family's ``predict``
                method.
            **kwargs (Any): Keyword arguments of the variational family's
                ``predict`` method.

        Returns
        -------
            GaussianDistribution: The output of the variational family's ``predict`` method.
        """
        raise NotImplementedError


@dataclass
class AbstractVariationalGaussian(AbstractVariationalFamily):
    r"""The variational Gaussian family of probability distributions."""

    inducing_inputs: Float[Array, "N D"]
    jitter: ScalarFloat = static_field(1e-6)

    @property
    def num_inducing(self) -> int:
        """The number of inducing inputs."""
        return self.inducing_inputs.shape[0]


@dataclass
class VariationalGaussian(AbstractVariationalGaussian):
    r"""The variational Gaussian family of probability distributions.

    The variational family is $`q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u`$, where
    $`u = f(z)`$ are the function values at the inducing inputs $`z`$
    and the distribution over the inducing inputs is
    $`q(u) = \mathcal{N}(\mu, S)`$.  We parameterise this over
    $`\mu`$ and $`sqrt`$ with $`S = sqrt sqrt^{\top}`$.
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

    def prior_kl(self) -> ScalarFloat:
        r"""Compute the prior KL divergence.

        Compute the KL-divergence between our variational approximation and the
        Gaussian process prior.

        For this variational family, we have
        ```math
        \begin{align}
        \operatorname{KL}[q(f(\cdot))\mid\mid p(\cdot)] & = \operatorname{KL}[q(u)\mid\mid p(u)]\\
        & = \operatorname{KL}[ \mathcal{N}(\mu, S) \mid\mid N(\mu z, \mathbf{K}_{zz}) ],
        \end{align}
        ```
        where $`u = f(z)`$ and $`z`$ are the inducing inputs.

        Returns
        -------
             ScalarFloat: The KL-divergence between our variational
                approximation and the GP prior.
        """
        # Unpack variational parameters
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        z = self.inducing_inputs

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        muz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz = cola.PSD(Kzz + cola.ops.I_like(Kzz) * self.jitter)

        sqrt = cola.ops.Triangular(sqrt)
        S = sqrt @ sqrt.T

        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.atleast_1d(muz.squeeze()), scale=Kzz)

        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs t.

        This is the integral $`q(f(t)) = \int p(f(t)\mid u) q(u) \mathrm{d}u`$, which
        can be computed in closed form as:
        ```math
            \mathcal{N}\left(f(t); \mu t + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} (\mu - \mu z),  \mathbf{K}_{tt} - \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt} + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} S \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt}\right).
        ```

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to
                make a prediction.

        Returns
        -------
            GaussianDistribution: The predictive distribution of the low-rank GP at
                the test inputs.
        """
        # Unpack variational parameters
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        z = self.inducing_inputs

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzz = kernel.gram(z)
        Kzz += cola.ops.I_like(Kzz) * self.jitter
        Lz = lower_cholesky(Kzz)
        muz = mean_function(z)

        # Unpack test inputs
        t = test_inputs

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = cola.solve(Lz, Kzt)

        # Kzz⁻¹ Kzt
        Kzz_inv_Kzt = cola.solve(Lz.T, Lz_inv_Kzt)

        # Ktz Kzz⁻¹ sqrt
        Ktz_Kzz_inv_sqrt = jnp.matmul(Kzz_inv_Kzt.T, sqrt)

        # μt + Ktz Kzz⁻¹ (μ - μz)
        mean = mut + jnp.matmul(Kzz_inv_Kzt.T, mu - muz)

        # Ktt - Ktz Kzz⁻¹ Kzt  +  Ktz Kzz⁻¹ S Kzz⁻¹ Kzt  [recall S = sqrt sqrtᵀ]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Kzz_inv_sqrt, Ktz_Kzz_inv_sqrt.T)
        )
        covariance += cola.ops.I_like(covariance) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


@dataclass
class WhitenedVariationalGaussian(VariationalGaussian):
    r"""The whitened variational Gaussian family of probability distributions.

    The variational family is $`q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u`$,
    where $`u = f(z)`$
    are the function values at the inducing inputs $`z`$ and the distribution over
    the inducing inputs is $`q(u) = \mathcal{N}(Lz \mu + mz, Lz S Lz^{\top})`$. We parameterise this
    over $`\mu`$ and $`sqrt`$ with $`S = sqrt sqrt^{\top}`$.
    """

    def prior_kl(self) -> ScalarFloat:
        r"""Compute the KL-divergence between our variational approximation and
        the Gaussian process prior.

        For this variational family, we have
        ```math
        \begin{align}
        \operatorname{KL}[q(f(\cdot))\mid\mid p(\cdot)] & = \operatorname{KL}[q(u)\mid\mid p(u)]\\
            & = \operatorname{KL}[N(\mu  , S)\mid\mid N(0, I)].
        \end{align}
        ```

        Returns
        -------
            ScalarFloat: The KL-divergence between our variational
                approximation and the GP prior.
        """
        # Unpack variational parameters
        mu = self.variational_mean
        sqrt = cola.ops.Triangular(self.variational_root_covariance)

        # S = LLᵀ
        S = sqrt @ sqrt.T

        # Compute whitened KL divergence
        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.zeros_like(jnp.atleast_1d(mu.squeeze())))
        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs t.

        This is the integral q(f(t)) = \int p(f(t)\midu) q(u) du, which can be computed in
        closed form as
        ```math
            \mathcal{N}\left(f(t); \mu t  +  \mathbf{K}_{tz} \mathbf{L}z^{\top} \mu  ,  \mathbf{K}_{tt}  -  \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt}  +  \mathbf{K}_{tz} \mathbf{L}z^{\top} S \mathbf{L}z^{-1} \mathbf{K}_{zt} \right).
        ```

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to
                make a prediction.

        Returns
        -------
            GaussianDistribution: The predictive distribution of the low-rank GP at
                the test inputs.
        """
        # Unpack variational parameters
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        z = self.inducing_inputs

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzz = kernel.gram(z)
        Kzz += cola.ops.I_like(Kzz) * self.jitter
        Lz = lower_cholesky(Kzz)

        # Unpack test inputs
        t = test_inputs

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = cola.solve(Lz, Kzt)

        # Ktz Lz⁻ᵀ sqrt
        Ktz_Lz_invT_sqrt = jnp.matmul(Lz_inv_Kzt.T, sqrt)

        # μt  +  Ktz Lz⁻ᵀ μ
        mean = mut + jnp.matmul(Lz_inv_Kzt.T, mu)

        # Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Lz⁻ᵀ S Lz⁻¹ Kzt  [recall S = sqrt sqrtᵀ]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Lz_invT_sqrt, Ktz_Lz_invT_sqrt.T)
        )
        covariance += cola.ops.I_like(covariance) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


@dataclass
class NaturalVariationalGaussian(AbstractVariationalGaussian):
    r"""The natural variational Gaussian family of probability distributions.

    The variational family is $`q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u`$,
    where $`u = f(z)`$ are
    the function values at the inducing inputs $`z`$ and the distribution over the
    inducing inputs is $`q(u) = N(\mu, S)`$. Expressing the variational distribution, in
    the form of the exponential family, $`q(u) = exp(\theta^{\top} T(u) - a(\theta))`$, gives rise to the
    natural parameterisation $`\theta  = (\theta_{1}, \theta_{2}) = (S^{-1}\mu, -S^{-1}/2)`$, to perform model inference,
    where $`T(u) = [u, uu^{\top}]`$ are the sufficient statistics.
    """

    natural_vector: Float[Array, "M 1"] = param_field(None, trainable=False)
    natural_matrix: Float[Array, "M M"] = param_field(None, trainable=False)

    def __post_init__(self):
        if self.natural_vector is None:
            self.natural_vector = jnp.zeros((self.num_inducing, 1))

        if self.natural_matrix is None:
            self.natural_matrix = -0.5 * jnp.eye(self.num_inducing)

    def prior_kl(self) -> ScalarFloat:
        r"""Compute the KL-divergence between our current variational approximation
        and the Gaussian process prior.

        For this variational family, we have
        ```math
        \begin{align}
        \operatorname{KL}[q(f(\cdot))\mid\mid p(\cdot)] & = \operatorname{KL}[q(u)\mid\mid p(u)] \\
            & = \operatorname{KL}[N(\mu, S)\mid\mid N(mz, \mathbf{K}_{zz})],
        \end{align}
        ```
        with $\mu$ and $S$ computed from the natural parameterisation $\theta  = (S^{-1}\mu  , -S^{-1}/2)$.

        Returns
        -------
            ScalarFloat: The KL-divergence between our variational approximation and
                the GP prior.
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
        sqrt = cola.ops.Triangular(sqrt)

        # S = LLᵀ:
        S = sqrt @ sqrt.T

        # μ = Sθ₁
        mu = S @ natural_vector

        muz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz += cola.ops.I_like(Kzz) * self.jitter

        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.atleast_1d(muz.squeeze()), scale=Kzz)

        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs $t$.

        This is the integral $`q(f(t)) = \int p(f(t)\mid u) q(u) \mathrm{d}u`$, which
        can be computed in closed form as
        ```math
             \mathcal{N}\left(f(t); \mu  t + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} (\mu   - \mu  z),  \mathbf{K}_{tt} - \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt} + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} S \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt} \right),
        ```
        with $`\mu`$ and $`S`$ computed from the natural parameterisation
        $`\theta = (S^{-1}\mu  , -S^{-1}/2)`$.

        Returns
        -------
            GaussianDistribution: A function that accepts a set of test points and will
                return the predictive distribution at those points.
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
        Kzz += cola.ops.I_like(Kzz) * self.jitter
        Lz = lower_cholesky(Kzz)
        muz = mean_function(z)

        Ktt = kernel.gram(test_inputs)
        Kzt = kernel.cross_covariance(z, test_inputs)
        mut = mean_function(test_inputs)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = cola.solve(Lz, Kzt)

        # Kzz⁻¹ Kzt
        Kzz_inv_Kzt = cola.solve(Lz.T, Lz_inv_Kzt)

        # Ktz Kzz⁻¹ L
        Ktz_Kzz_inv_L = jnp.matmul(Kzz_inv_Kzt.T, sqrt)

        # μt  +  Ktz Kzz⁻¹ (μ  -  μz)
        mean = mut + jnp.matmul(Kzz_inv_Kzt.T, mu - muz)

        # Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Kzz⁻¹ S Kzz⁻¹ Kzt  [recall S = LLᵀ]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Kzz_inv_L, Ktz_Kzz_inv_L.T)
        )
        covariance += cola.ops.I_like(covariance) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )

    def to_expectation(self) -> "ExpectationVariationalGaussian":
        natural_matrix = self.natural_matrix
        natural_vector = self.natural_vector

        # S = (-2θ₂)⁻¹
        S_inv = -2.0 * natural_matrix
        S_inv += cola.ops.I_like(S_inv) * self.jitter
        S_inv = cola.PSD(S_inv)
        S = cola.inverse(S_inv)

        # μ = Sθ₁
        mu = S @ natural_vector

        # η₁ = μ
        expectation_vector = mu

        # η₂ = S + μ μᵀ
        expectation_matrix = S.to_dense() + jnp.matmul(mu, mu.T)

        return ExpectationVariationalGaussian(
            posterior=self.posterior,
            inducing_inputs=self.inducing_inputs,
            expectation_vector=expectation_vector,
            expectation_matrix=expectation_matrix,
        )


@dataclass
class ExpectationVariationalGaussian(AbstractVariationalGaussian):
    r"""The natural variational Gaussian family of probability distributions.

    The variational family is $`q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u`$, where $`u = f(z)`$ are the
    function values at the inducing inputs $`z`$ and the distribution over the inducing
    inputs is $`q(u) = \mathcal{N}(\mu, S)`$. Expressing the variational distribution, in the form of
    the exponential family, $`q(u) = exp(\theta^{\top} T(u) - a(\theta))`$, gives rise to the natural
    parameterisation $`\theta  = (\theta_{1}, \theta_{2}) = (S^{-1}\mu  , -S^{-1}/2)`$ and sufficient statistics
    $`T(u) = [u, uu^{\top}]`$. The expectation parameters are given by $`\nu = \int T(u) q(u) \mathrm{d}u`$.
    This gives a parameterisation, $`\nu = (\nu_{1}, \nu_{2}) = (\mu  , S + uu^{\top})`$ to perform model
    inference over.
    """

    expectation_vector: Float[Array, "M 1"] = None
    expectation_matrix: Float[Array, "M M"] = None

    def __post_init__(self):
        if self.expectation_vector is None:
            self.expectation_vector = jnp.zeros((self.num_inducing, 1))
        if self.expectation_matrix is None:
            self.expectation_matrix = jnp.eye(self.num_inducing)

    def prior_kl(self) -> ScalarFloat:
        r"""Evaluate the prior KL-divergence.

        Compute the KL-divergence between our current variational approximation and
        the Gaussian process prior.

        For this variational family, we have
        ```math
        \begin{align}
        \operatorname{KL}(q(f(\cdot))\mid\mid p(\cdot)) & = \operatorname{KL}(q(u)\mid\mid p(u)) \\
            & =\operatorname{KL}(\mathcal{N}(\mu, S)\mid\mid \mathcal{N}(m_z, K_{zz})),
        \end{align}
        ```
        where $\mu$ and $S$ are the expectation parameters of the variational
        distribution and $m_z$ and $K_{zz}$ are the mean and covariance of the prior
        distribution.

        Returns
        -------
            ScalarFloat: The KL-divergence between our variational approximation and
                the GP prior.
        """
        # Unpack variational parameters
        expectation_vector = self.expectation_vector
        expectation_matrix = self.expectation_matrix
        z = self.inducing_inputs

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # μ = η₁
        mu = expectation_vector

        # S = η₂ - η₁ η₁ᵀ
        S = expectation_matrix - jnp.outer(mu, mu)
        S = cola.ops.Dense(S)
        S += cola.ops.I_like(S) * self.jitter
        S = cola.PSD(S)

        muz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz += cola.ops.I_like(Kzz) * self.jitter

        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.atleast_1d(muz.squeeze()), scale=Kzz)

        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        r"""Evaluate the predictive distribution.

        Compute the predictive distribution of the GP at the test inputs $t$.

        This is the integral $q(f(t)) = \int p(f(t)\mid u)q(u)\mathrm{d}u$, which can
        be computed in closed form as  which can be computed in closed form as
        ```math
        \mathcal{N}(f(t); \mu_t + \mathbf{K}_{tz}\mathbf{K}_{zz}^{-1}(\mu - \mu_z), \mathbf{K}_{tt} - \mathbf{K}_{tz}\mathbf{K}_{zz}^{-1}\mathbf{K}_{zt} + \mathbf{K}_{tz}\mathbf{K}_{zz}^{-1}\mathbf{S} \mathbf{K}_{zz}^{-1}\mathbf{K}_{zt})
        ```

        with $\mu$ and $S$ computed from the expectation parameterisation
        $\eta = (\mu, S + uu^\top)$.

        Returns
        -------
            GaussianDistribution: The predictive distribution of the GP at the
                test inputs $t$.
        """
        # Unpack variational parameters
        expectation_vector = self.expectation_vector
        expectation_matrix = self.expectation_matrix
        z = self.inducing_inputs

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # μ = η₁
        mu = expectation_vector

        # S = η₂ - η₁ η₁ᵀ
        S = expectation_matrix - jnp.matmul(mu, mu.T)
        S = cola.ops.Dense(S)
        S += cola.ops.I_like(S) * self.jitter
        S = cola.PSD(S)

        # S = sqrt sqrtᵀ
        sqrt = lower_cholesky(S)

        Kzz = kernel.gram(z)
        Kzz += cola.ops.I_like(Kzz) * self.jitter
        Lz = lower_cholesky(Kzz)
        muz = mean_function(z)

        # Unpack test inputs
        t = test_inputs

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = cola.solve(Lz, Kzt)

        # Kzz⁻¹ Kzt
        Kzz_inv_Kzt = cola.solve(Lz.T, Lz_inv_Kzt)

        # Ktz Kzz⁻¹ sqrt
        Ktz_Kzz_inv_sqrt = Kzz_inv_Kzt.T @ sqrt

        # μt  +  Ktz Kzz⁻¹ (μ  -  μz)
        mean = mut + jnp.matmul(Kzz_inv_Kzt.T, mu - muz)

        # Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Kzz⁻¹ S Kzz⁻¹ Kzt  [recall S = sqrt sqrtᵀ]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Kzz_inv_sqrt, Ktz_Kzz_inv_sqrt.T)
        )
        covariance += cola.ops.I_like(covariance) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )

    def to_natural(self) -> "NaturalVariationalGaussian":
        expectation_matrix = self.expectation_matrix
        expectation_vector = self.expectation_vector

        # S = η₂ - η₁ η₁ᵀ
        S = cola.PSD(
            cola.ops.Dense(expectation_matrix)
            - jnp.matmul(expectation_vector, expectation_vector.T)
        )

        # θ₂ = -S⁻¹/2
        S += cola.ops.I_like(S) * self.jitter
        S_inv = cola.inverse(S)
        natural_matrix = -0.5 * S_inv.to_dense()

        # θ₁ = S⁻¹μ = S⁻¹η₁
        natural_vector = S_inv @ expectation_vector

        return NaturalVariationalGaussian(
            posterior=self.posterior,
            inducing_inputs=self.inducing_inputs,
            natural_vector=natural_vector,
            natural_matrix=natural_matrix,
        )


@dataclass
class DualVariationalGaussian(VariationalGaussian):
    """The Dual variational parameterisation. The Dual vector and matrix correspond to the learnt sufficient statistics of the learnt pseudo likelihood."""

    dual_vector: Float[Array, "M 1"] = param_field(None, trainable=False)
    dual_matrix: Float[Array, "M M"] = param_field(None, trainable=False)

    def __post_init__(self):
        if self.dual_vector is None:
            self.dual_vector = jnp.zeros((self.num_inducing, 1))

        if self.dual_matrix is None:
            self.dual_matrix = -jnp.eye(self.num_inducing) * 1e-6

    def pseudo_data(self) -> Tuple[Float[Array, "M 1"], Float[Array, "M M"]]:
        """Returns the pseudo Gaussian observations, $\tilde{y}$, and corresponding covariance, $\tilde{\\Sigma}$, i.e., $\tilde{y} \\sim \\mathcal{N}((f(z), \tilde{\\Sigma})$.

        Returns
        -------
            Tuple[Float[Array, "M 1"], Float[Array, "M M"]]: The pseudo data y and pseudo covariance.
        """
        # Unpack variational parameters
        dual_vector, dual_matrix = self.dual_vector, self.dual_matrix

        # cov = (-2λ₂)⁻¹
        inv_peusdo_cov = -2.0 * cola.ops.Dense(dual_matrix)
        inv_peusdo_cov += cola.ops.I_like(inv_peusdo_cov) * self.posterior.prior.jitter
        inv_peusdo_cov = cola.PSD(inv_peusdo_cov)
        peusdo_cov = cola.inverse(inv_peusdo_cov)

        # y = (-2λ₂)⁻¹λ₁
        peusdo_y = peusdo_cov @ dual_vector

        return peusdo_y, peusdo_cov

    # This is the ConjugateLikelihood at the inducing inputs z, but with multivariate likelihood cov.
    # TODO: Create this as an objective like conjugate MLL + add Multivate noise cov Gaussian likelihood.
    def pseudo_log_likelihood(self) -> ScalarFloat:
        """Returns the log likelihood of the pseudo data, $\tilde{y}$, under the pseudo likelihood.

            $$ \\int p(f(z))\\mathcal{N}(\tilde{y}; f(z),  \tilde{\\Sigma}) $$

            This is the same as marginal likelihood for conjugate observations $\tilde{y} \\sim \\mathcal{N}(f(z), \tilde{\\Sigma})$.

        Returns
        -------
            ScalarFloat: The log likelihood of the pseudo data under the pseudo likelihood.
        """
        posterior = self.posterior
        z = self.inducing_inputs

        # Compute the pseudo data from the dual parameters.
        y, cov = self.pseudo_data()

        # Compute the marginal likelihood of the pseudo data under the prior.
        mz = posterior.prior.mean_function(z)
        Kzz = posterior.prior.kernel.gram(z)
        Kzz += cola.ops.I_like(Kzz) * posterior.prior.jitter
        mll = GaussianDistribution(jnp.atleast_1d(mz.squeeze()), Kzz + cov)

        return mll.log_prob(jnp.atleast_1d(y.squeeze())).sum()

    # This is the AnalyticalGaussianIntegrator at the inducing inputs z, but with multivariate likelihood S.
    # TODO: extend Gaussian and AnalyticalGaussianIntegrator to multivariate likelihoods.
    def peusdo_expected_density(self) -> ScalarFloat:
        """Returns the expected density of the pseudo data. This is the same as the AnalyticalGaussianIntegrator at the inducing inputs z for the pseudo data.

            $$ \\int q(f(z))\\log(\\mathcal{N}(\tilde{y}; f(z),  \tilde{\\Sigma})) $$

        Returns
        -------
            ScalarFloat: The expected density of the pseudo data.
        """
        z = self.inducing_inputs
        qz = self.predict(z)
        mu, S = qz.loc, qz.scale

        # Compute the pseudo data from the dual parameters.
        pseudo_y, pseudo_cov = self.pseudo_data()

        # Compute the marginal likelihood of the pseudo data under the prior.
        ml = (
            GaussianDistribution(
                jnp.atleast_1d(mu.squeeze()),
                pseudo_cov + cola.ops.I_like(pseudo_cov) * self.posterior.prior.jitter,
            )
            .log_prob(jnp.atleast_1d(pseudo_y.squeeze()))
            .sum()
        )
        trace = -0.5 * jnp.trace(
            (
                cola.inverse(
                    cola.PSD(
                        pseudo_cov
                        + cola.ops.I_like(pseudo_cov) * self.posterior.prior.jitter
                    )
                )
                @ S
            ).to_dense()
        )
        return ml + trace

    def prior_kl(self) -> ScalarFloat:
        r"""Evaluate the prior KL-divergence.

        Compute the KL-divergence between our current variational approximation and the Gaussian process prior.

        For this variational family, we have

        ```math
        \begin{align}
        \operatorname{KL}(q(f(\cdot))\mid\mid p(\cdot)) & = \int q(f(z))\log(\mathcal{N}(\tilde{y}; f(z),  \tilde{\Sigma})) - \int p(f(z))\mathcal{N}(\tilde{y}; f(z),  \tilde{\Sigma})
        \end{align}
        ```
        where $\mu$ and $S$ are the expectation parameters of the variational distribution and $m_z$ and $K_{zz}$ are the mean and covariance of the prior distribution.

        Returns
        -------
            ScalarFloat: The KL-divergence between our variational approximation and the GP prior.
        """
        return self.peusdo_expected_density() - self.pseudo_log_likelihood()

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs t.

        This is the integral $`q(f(t)) = \int p(f(t)\mid u) q(u) \mathrm{d}u`$, which
        can be computed in closed form as:
        ```math
            \mathcal{N}\left(f(t); \mu(t) + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} (\tilde{y} - \mu(z)),  \mathbf{K}_{tt} - \mathbf{K}_{tz} (\mathbf{K}_{zz} + \tilde{\Sigma})^{-1} \mathbf{K}_{zt} \right).
        ```

        Where $\tilde{y}$ and $\tilde{\Sigma}$ are computed from the dual parameterisation $(\lambda_1, \lambda_2) = (\tilde{\Sigma}^{-1}\tilde{y}, -\tilde{\Sigma}^{-1}/2 = (-2 )^{-1})$.

        Observe this is the same as conditioning a the Gaussian process prior on the pseudo Gaussian obvserions, $\tilde{y} \sim \mathcal{N}((f(z), \tilde{\Sigma})$.

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to
                make a prediction.

        Returns
        -------
            GaussianDistribution: The predictive distribution of the low-rank GP at
                the test inputs.
        """

        # These are same as conjugate Gaussian process prior - where we subsitite in the pseudo data.
        # The issue with just using our current one in GPJax - is that we don't support multivariate noise likelihoods.

        # Alternatively could add the natural parameters of the pseudo likelihood that correspond to the dual parameters, to the natural parameters of the prior
        # And use NaturalVariationalGaussian's predict.

        # Regardless, here is some code that works for now.

        # Unpack variational parameters
        pseudo_y, pseudo_cov = self.pseudo_data()
        z = self.inducing_inputs

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzz = kernel.gram(z)
        Kzz += cola.ops.I_like(Kzz) * self.jitter
        mz = mean_function(z)

        # Unpack test inputs
        t = test_inputs

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        mt = mean_function(t)

        # (Kzz + pseudo_cov)
        Sigma = cola.PSD(Kzz + pseudo_cov)

        # (Kzz + pseudo_cov)⁻¹ Kzt
        Sigma_inv_Kzt = cola.solve(Sigma, Kzt)

        # mt + Ktz (Kzz + pseudo_cov)⁻¹ (pseudo_y - mz)
        mean = mt + Sigma_inv_Kzt.T @ (pseudo_y - mz)

        # Ktt -  Ktz (Kzz + pseudo_cov)⁻¹ Kzt
        covariance = Ktt - Sigma_inv_Kzt.T @ Kzt
        covariance += cola.ops.I_like(covariance) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )

    # TODO: This just gives you an idea of how to compute the natural gradient.
    # We need to work out some API
    # It would be nice not to reuse all the variational expectation code.
    # (This was done so due to squeezing a vmap, to facilitate grad on (1, ) array).
    def natural_gradient(self, train_data: Dataset) -> "DualVariationalGaussian":
        r"""Compute the natural_gradient.

        Returns
        -------
            DualVariationalGaussian: The natural gradient of the variational parameters.
        """

        ## --- REUSING CODE FROM EXPECTATION VARIATIONAL GAUSSIAN ---

        # Unpack training batch
        x, y = train_data.X, train_data.y

        # Variational distribution q(f(·)) = N(f(·); μ(·), Σ(·, ·))
        q = self

        # TODO: This needs cleaning up! We are squeezing then broadcasting `mean` and `variance`, which is not ideal.

        # Compute variational mean, μ(x), and variance, diag(Σ(x, x)), at the training
        # inputs, x
        def q_moments(x):
            qx = q(x)
            return qx.mean().squeeze(), qx.covariance().squeeze()

        mean, variance = vmap(q_moments)(x[:, None])

        # ≈ ∫[log(p(y|f(x))) q(f(x))] df(x)
        exp_ll = lambda y, m, v: q.posterior.likelihood.expected_log_likelihood(
            y, m, v
        ).squeeze()  # <---- CHANGED THIS

        _, (dL_mi, dL_vi) = vmap(
            value_and_grad(exp_ll, argnums=(1, 2)), in_axes=(0, 0, 0)
        )(
            y[:, None], mean[:, None], variance[:, None]
        )  # <---- CHANGED THIS

        ## --- REUSING CODE FROM EXPECTATION VARIATIONAL GAUSSIAN ---

        # Clip gradients to ensure negative definiteness of the dual matrix
        dL_vi = jnp.minimum(dL_vi, -1e-12 * jnp.ones_like(dL_vi))

        # These are natural gradients at each dual site q(f(xi)):
        dL_mu_1i = dL_mi - 2.0 * dL_vi * mean[:, None]
        dL_mu_2i = dL_vi

        # Now we compute the project these down to compute natural gradients of the low-rank dual variational parameters
        z = self.inducing_inputs
        Kzz = self.posterior.prior.kernel.gram(z)
        Kzx = self.posterior.prior.kernel.cross_covariance(z, x)
        Kzz_inv_Kzx = (
            cola.inverse(
                cola.PSD(Kzz + cola.ops.I_like(Kzz) * self.posterior.prior.jitter)
            )
            @ Kzx
        )

        natgrad_dual_vector = Kzz_inv_Kzx @ dL_mu_1i
        natgrad_dual_matrix = (
            Kzz_inv_Kzx @ cola.ops.Diagonal(dL_mu_2i.squeeze()) @ Kzz_inv_Kzx.T
        )

        return self.replace(
            dual_vector=natgrad_dual_vector, dual_matrix=natgrad_dual_matrix
        )


@dataclass
class CollapsedVariationalGaussian(AbstractVariationalGaussian):
    r"""Collapsed variational Gaussian.

    Collapsed variational Gaussian family of probability distributions.
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables
    in Sparse Gaussian Processes.
    """

    def __post_init__(self):
        if not isinstance(self.posterior.likelihood, Gaussian):
            raise TypeError("Likelihood must be Gaussian.")

    def predict(
        self, test_inputs: Float[Array, "N D"], train_data: Dataset
    ) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs.

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs $t$ at which to make
                predictions.
            train_data (Dataset): The training data that was used to fit the GP.

        Returns
        -------
            GaussianDistribution: The predictive distribution of the collapsed
                variational Gaussian process at the test inputs $t$.
        """
        # Unpack test inputs
        t = test_inputs

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
        Kzz += cola.ops.I_like(Kzz) * self.jitter

        # Lz Lzᵀ = Kzz
        Lz = lower_cholesky(Kzz)

        # Lz⁻¹ Kzx
        Lz_inv_Kzx = cola.solve(Lz, Kzx)

        # A = Lz⁻¹ Kzt / o
        A = Lz_inv_Kzx / jnp.sqrt(noise)

        # AAᵀ
        AAT = jnp.matmul(A, A.T)

        # LLᵀ = I + AAᵀ
        L = jnp.linalg.cholesky(jnp.eye(m) + AAT)

        mux = mean_function(x)
        diff = y - mux

        # Lz⁻¹ Kzx (y - μx)
        Lz_inv_Kzx_diff = jsp.linalg.cho_solve((L, True), jnp.matmul(Lz_inv_Kzx, diff))

        # Kzz⁻¹ Kzx (y - μx)
        Kzz_inv_Kzx_diff = cola.solve(Lz.T, Lz_inv_Kzx_diff)

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = cola.solve(Lz, Kzt)

        # L⁻¹ Lz⁻¹ Kzt
        L_inv_Lz_inv_Kzt = jsp.linalg.solve_triangular(L, Lz_inv_Kzt, lower=True)

        # μt + 1/o² Ktz Kzz⁻¹ Kzx (y - μx)
        mean = mut + jnp.matmul(Kzt.T / noise, Kzz_inv_Kzx_diff)

        # Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Lz⁻¹ (I + AAᵀ)⁻¹ Lz⁻¹ Kzt
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(L_inv_Lz_inv_Kzt.T, L_inv_Lz_inv_Kzt)
        )
        covariance += cola.ops.I_like(covariance) * self.jitter

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
    "DualVariationalGaussian",
    "CollapsedVariationalGaussian",
]
