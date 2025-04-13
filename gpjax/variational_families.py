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

import beartype.typing as tp
from cola.annotations import PSD
from cola.linalg.decompositions.decompositions import Cholesky
from cola.linalg.inverse.inv import solve
from cola.ops.operators import (
    Dense,
    I_like,
    Identity,
    Triangular,
)
from flax import nnx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.gps import (
    AbstractPosterior,
    AbstractPrior,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import (
    Gaussian,
    NonGaussian,
)
from gpjax.lower_cholesky import lower_cholesky
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.parameters import (
    LowerTriangular,
    Real,
    Static,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)

K = tp.TypeVar("K", bound=AbstractKernel)
M = tp.TypeVar("M", bound=AbstractMeanFunction)
L = tp.TypeVar("L", Gaussian, NonGaussian)
NGL = tp.TypeVar("NGL", bound=NonGaussian)
GL = tp.TypeVar("GL", bound=Gaussian)
P = tp.TypeVar("P", bound=AbstractPrior)
PP = tp.TypeVar("PP", bound=AbstractPosterior)


class AbstractVariationalFamily(nnx.Module, tp.Generic[L]):
    r"""
    Abstract base class used to represent families of distributions that can be
    used within variational inference.
    """

    def __init__(self, posterior: AbstractPosterior[P, L]):
        self.posterior = posterior

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> GaussianDistribution:
        r"""Evaluate the variational family's density.

        For a given set of parameters, compute the latent function's prediction
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
    def predict(self, *args: tp.Any, **kwargs: tp.Any) -> GaussianDistribution:
        r"""Predict the GP's output given the input.

        Args:
            *args (Any): Arguments of the variational family's ``predict``
                method.
            **kwargs (Any): Keyword arguments of the variational family's
                ``predict`` method.

        Returns:
            GaussianDistribution: The output of the variational family's ``predict`` method.
        """
        raise NotImplementedError


class AbstractVariationalGaussian(AbstractVariationalFamily[L]):
    r"""The variational Gaussian family of probability distributions."""

    def __init__(
        self,
        posterior: AbstractPosterior[P, L],
        inducing_inputs: tp.Union[
            Float[Array, "N D"],
            Real,
            Static,
        ],
        jitter: ScalarFloat = 1e-6,
    ):
        if not isinstance(inducing_inputs, (Real, Static)):
            inducing_inputs = Real(inducing_inputs)

        self.inducing_inputs = inducing_inputs
        self.jitter = jitter

        super().__init__(posterior)

    @property
    def num_inducing(self) -> int:
        """The number of inducing inputs."""
        return self.inducing_inputs.value.shape[0]


class VariationalGaussian(AbstractVariationalGaussian[L]):
    r"""The variational Gaussian family of probability distributions.

    The variational family is $q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u$, where
    $u = f(z)$ are the function values at the inducing inputs $z$
    and the distribution over the inducing inputs is
    $q(u) = \mathcal{N}(\mu, S)$.  We parameterise this over
    $\mu$ and $sqrt$ with $S = sqrt sqrt^{\top}$.
    """

    def __init__(
        self,
        posterior: AbstractPosterior[P, L],
        inducing_inputs: Float[Array, "N D"],
        variational_mean: tp.Union[Float[Array, "N 1"], None] = None,
        variational_root_covariance: tp.Union[Float[Array, "N N"], None] = None,
        jitter: ScalarFloat = 1e-6,
    ):
        super().__init__(posterior, inducing_inputs, jitter)

        if variational_mean is None:
            variational_mean = jnp.zeros((self.num_inducing, 1))

        if variational_root_covariance is None:
            variational_root_covariance = jnp.eye(self.num_inducing)

        self.variational_mean = Real(variational_mean)
        self.variational_root_covariance = LowerTriangular(variational_root_covariance)

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
        where $u = f(z)$ and $z$ are the inducing inputs.

        Returns:
            ScalarFloat: The KL-divergence between our variational
                approximation and the GP prior.
        """
        # Unpack variational parameters
        mu = self.variational_mean.value
        sqrt = self.variational_root_covariance.value
        z = self.inducing_inputs.value

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        muz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz = PSD(Kzz + I_like(Kzz) * self.jitter)

        sqrt = Triangular(sqrt)
        S = sqrt @ sqrt.T

        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.atleast_1d(muz.squeeze()), scale=Kzz)

        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs t.

        This is the integral $q(f(t)) = \int p(f(t)\mid u) q(u) \mathrm{d}u$, which
        can be computed in closed form as:
        ```math
            \mathcal{N}\left(f(t); \mu t + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} (\mu - \mu z),  \mathbf{K}_{tt} - \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt} + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} S \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt}\right).
        ```

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to
                make a prediction.

        Returns:
            GaussianDistribution: The predictive distribution of the low-rank GP at
                the test inputs.
        """
        # Unpack variational parameters
        mu = self.variational_mean.value
        sqrt = self.variational_root_covariance.value
        z = self.inducing_inputs.value

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzz = kernel.gram(z)
        Kzz += I_like(Kzz) * self.jitter
        Lz = lower_cholesky(Kzz)
        muz = mean_function(z)

        # Unpack test inputs
        t = test_inputs

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = solve(Lz, Kzt, Cholesky())

        # Kzz⁻¹ Kzt
        Kzz_inv_Kzt = solve(Lz.T, Lz_inv_Kzt, Cholesky())

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
        covariance += I_like(covariance) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


class WhitenedVariationalGaussian(VariationalGaussian[L]):
    r"""The whitened variational Gaussian family of probability distributions.

    The variational family is $q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u$,
    where $u = f(z)$
    are the function values at the inducing inputs $z$ and the distribution over
    the inducing inputs is $q(u) = \mathcal{N}(Lz \mu + mz, Lz S Lz^{\top})$. We parameterise this
    over $\mu$ and $sqrt$ with $S = sqrt sqrt^{\top}$.
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

        Returns:
            ScalarFloat: The KL-divergence between our variational
                approximation and the GP prior.
        """
        # Unpack variational parameters
        mu = self.variational_mean.value
        sqrt = Triangular(self.variational_root_covariance.value)

        # S = LLᵀ
        S = sqrt @ sqrt.T

        # Compute whitened KL divergence
        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu_S = Identity(shape=(self.num_inducing, self.num_inducing), dtype=mu.dtype)
        pu = GaussianDistribution(
            loc=jnp.zeros_like(jnp.atleast_1d(mu.squeeze())), scale=pu_S
        )
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

        Returns:
            GaussianDistribution: The predictive distribution of the low-rank GP at
                the test inputs.
        """
        # Unpack variational parameters
        mu = self.variational_mean.value
        sqrt = self.variational_root_covariance.value
        z = self.inducing_inputs.value

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzz = kernel.gram(z)
        Kzz += I_like(Kzz) * self.jitter
        Lz = lower_cholesky(Kzz)

        # Unpack test inputs
        t = test_inputs

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = solve(Lz, Kzt, Cholesky())

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
        covariance += I_like(covariance) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


class NaturalVariationalGaussian(AbstractVariationalGaussian[L]):
    r"""The natural variational Gaussian family of probability distributions.

    The variational family is $q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u$,
    where $u = f(z)$ are
    the function values at the inducing inputs $z$ and the distribution over the
    inducing inputs is $q(u) = N(\mu, S)$. Expressing the variational distribution, in
    the form of the exponential family, $q(u) = exp(\theta^{\top} T(u) - a(\theta))$, gives rise to the
    natural parameterisation $\theta  = (\theta_{1}, \theta_{2}) = (S^{-1}\mu, -S^{-1}/2)$, to perform model inference,
    where $T(u) = [u, uu^{\top}]$ are the sufficient statistics.
    """

    def __init__(
        self,
        posterior: AbstractPosterior[P, L],
        inducing_inputs: Float[Array, "N D"],
        natural_vector: tp.Union[Float[Array, "M 1"], None] = None,
        natural_matrix: tp.Union[Float[Array, "M M"], None] = None,
        jitter: ScalarFloat = 1e-6,
    ):
        super().__init__(posterior, inducing_inputs, jitter)

        if natural_vector is None:
            natural_vector = jnp.zeros((self.num_inducing, 1))

        if natural_matrix is None:
            natural_matrix = -0.5 * jnp.eye(self.num_inducing)

        self.natural_vector = Static(natural_vector)
        self.natural_matrix = Static(natural_matrix)

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

        Returns:
            ScalarFloat: The KL-divergence between our variational approximation and
                the GP prior.
        """
        # Unpack variational parameters
        natural_vector = self.natural_vector.value
        natural_matrix = self.natural_matrix.value
        z = self.inducing_inputs.value
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
        sqrt = Triangular(sqrt)

        # S = LLᵀ:
        S = sqrt @ sqrt.T

        # μ = Sθ₁
        mu = S @ natural_vector

        muz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz += I_like(Kzz) * self.jitter

        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.atleast_1d(muz.squeeze()), scale=Kzz)

        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs $t$.

        This is the integral $q(f(t)) = \int p(f(t)\mid u) q(u) \mathrm{d}u$, which
        can be computed in closed form as
        ```math
             \mathcal{N}\left(f(t); \mu  t + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} (\mu   - \mu  z),  \mathbf{K}_{tt} - \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt} + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} S \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt} \right),
        ```
        with $\mu$ and $S$ computed from the natural parameterisation
        $\theta = (S^{-1}\mu  , -S^{-1}/2)$.

        Returns:
            GaussianDistribution: A function that accepts a set of test points and will
                return the predictive distribution at those points.
        """
        # Unpack variational parameters
        natural_vector = self.natural_vector.value
        natural_matrix = self.natural_matrix.value
        z = self.inducing_inputs.value
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
        Kzz += I_like(Kzz) * self.jitter
        Lz = lower_cholesky(Kzz)
        muz = mean_function(z)

        Ktt = kernel.gram(test_inputs)
        Kzt = kernel.cross_covariance(z, test_inputs)
        mut = mean_function(test_inputs)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = solve(Lz, Kzt, Cholesky())

        # Kzz⁻¹ Kzt
        Kzz_inv_Kzt = solve(Lz.T, Lz_inv_Kzt, Cholesky())

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
        covariance += I_like(covariance) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


class ExpectationVariationalGaussian(AbstractVariationalGaussian[L]):
    r"""The natural variational Gaussian family of probability distributions.

    The variational family is $q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u$, where $u = f(z)$ are the
    function values at the inducing inputs $z$ and the distribution over the inducing
    inputs is $q(u) = \mathcal{N}(\mu, S)$. Expressing the variational distribution, in the form of
    the exponential family, $q(u) = exp(\theta^{\top} T(u) - a(\theta))$, gives rise to the natural
    parameterisation $\theta  = (\theta_{1}, \theta_{2}) = (S^{-1}\mu  , -S^{-1}/2)$ and sufficient statistics
    $T(u) = [u, uu^{\top}]$. The expectation parameters are given by $\nu = \int T(u) q(u) \mathrm{d}u$.
    This gives a parameterisation, $\nu = (\nu_{1}, \nu_{2}) = (\mu  , S + uu^{\top})$ to perform model
    inference over.
    """

    def __init__(
        self,
        posterior: AbstractPosterior[P, L],
        inducing_inputs: Float[Array, "N D"],
        expectation_vector: tp.Union[Float[Array, "M 1"], None] = None,
        expectation_matrix: tp.Union[Float[Array, "M M"], None] = None,
        jitter: ScalarFloat = 1e-6,
    ):
        super().__init__(posterior, inducing_inputs, jitter)

        if expectation_vector is None:
            expectation_vector = jnp.zeros((self.num_inducing, 1))

        if expectation_matrix is None:
            expectation_matrix = jnp.eye(self.num_inducing)

        self.expectation_vector = Static(expectation_vector)
        self.expectation_matrix = Static(expectation_matrix)

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

        Returns:
            ScalarFloat: The KL-divergence between our variational approximation and
                the GP prior.
        """
        # Unpack variational parameters
        expectation_vector = self.expectation_vector.value
        expectation_matrix = self.expectation_matrix.value
        z = self.inducing_inputs.value

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # μ = η₁
        mu = expectation_vector

        # S = η₂ - η₁ η₁ᵀ
        S = expectation_matrix - jnp.outer(mu, mu)
        S = PSD(Dense(S))
        S += I_like(S) * self.jitter

        muz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz += I_like(Kzz) * self.jitter

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

        Returns:
            GaussianDistribution: The predictive distribution of the GP at the
                test inputs $t$.
        """
        # Unpack variational parameters
        expectation_vector = self.expectation_vector.value
        expectation_matrix = self.expectation_matrix.value
        z = self.inducing_inputs.value

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # μ = η₁
        mu = expectation_vector

        # S = η₂ - η₁ η₁ᵀ
        S = expectation_matrix - jnp.matmul(mu, mu.T)
        S = Dense(S) + I_like(S) * self.jitter
        S = PSD(S)

        # S = sqrt sqrtᵀ
        sqrt = lower_cholesky(S)

        Kzz = kernel.gram(z)
        Kzz += I_like(Kzz) * self.jitter
        Lz = lower_cholesky(Kzz)
        muz = mean_function(z)

        # Unpack test inputs
        t = test_inputs

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = solve(Lz, Kzt, Cholesky())

        # Kzz⁻¹ Kzt
        Kzz_inv_Kzt = solve(Lz.T, Lz_inv_Kzt, Cholesky())

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
        covariance += I_like(covariance) * self.jitter

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )


class CollapsedVariationalGaussian(AbstractVariationalGaussian[GL]):
    r"""Collapsed variational Gaussian.

    Collapsed variational Gaussian family of probability distributions.
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables
    in Sparse Gaussian Processes.
    """

    def __init__(
        self,
        posterior: AbstractPosterior[P, GL],
        inducing_inputs: Float[Array, "N D"],
        jitter: ScalarFloat = 1e-6,
    ):
        super().__init__(posterior, inducing_inputs, jitter)

        if not isinstance(posterior.likelihood, Gaussian):
            raise TypeError("Likelihood must be Gaussian.")

    def predict(
        self, test_inputs: Float[Array, "N D"], train_data: Dataset
    ) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs.

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs $t$ at which to make
                predictions.
            train_data (Dataset): The training data that was used to fit the GP.

        Returns:
            GaussianDistribution: The predictive distribution of the collapsed
                variational Gaussian process at the test inputs $t$.
        """
        # Unpack test inputs
        t = test_inputs

        # Unpack training data
        x, y = train_data.X, train_data.y

        # Unpack variational parameters
        noise_var = self.posterior.likelihood.obs_stddev.value**2
        z = self.inducing_inputs.value
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzx = kernel.cross_covariance(z, x)
        Kzz = kernel.gram(z)
        Kzz += I_like(Kzz) * self.jitter

        # Lz Lzᵀ = Kzz
        Lz = lower_cholesky(Kzz)

        # Lz⁻¹ Kzx
        Lz_inv_Kzx = solve(Lz, Kzx, Cholesky())

        # A = Lz⁻¹ Kzt / o
        A = Lz_inv_Kzx / self.posterior.likelihood.obs_stddev.value

        # AAᵀ
        AAT = jnp.matmul(A, A.T)

        # LLᵀ = I + AAᵀ
        L = jnp.linalg.cholesky(jnp.eye(m) + AAT)

        mux = mean_function(x)
        diff = y - mux

        # Lz⁻¹ Kzx (y - μx)
        Lz_inv_Kzx_diff = jsp.linalg.cho_solve((L, True), jnp.matmul(Lz_inv_Kzx, diff))

        # Kzz⁻¹ Kzx (y - μx)
        Kzz_inv_Kzx_diff = solve(Lz.T, Lz_inv_Kzx_diff, Cholesky())

        Ktt = kernel.gram(t)
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz⁻¹ Kzt
        Lz_inv_Kzt = solve(Lz, Kzt, Cholesky())

        # L⁻¹ Lz⁻¹ Kzt
        L_inv_Lz_inv_Kzt = jsp.linalg.solve_triangular(L, Lz_inv_Kzt, lower=True)

        # μt + 1/o² Ktz Kzz⁻¹ Kzx (y - μx)
        mean = mut + jnp.matmul(Kzt.T / noise_var, Kzz_inv_Kzx_diff)

        # Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Lz⁻¹ (I + AAᵀ)⁻¹ Lz⁻¹ Kzt
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(L_inv_Lz_inv_Kzt.T, L_inv_Lz_inv_Kzt)
        )
        covariance += I_like(covariance) * self.jitter

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
