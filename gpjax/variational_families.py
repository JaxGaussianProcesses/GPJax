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
from typing import Any, Callable, Dict, Optional

import distrax as dx
import jax.numpy as jnp
import jax.scipy as jsp
from chex import dataclass
from jaxtyping import Array, Float

from .config import get_defaults
from .covariance_operator import I
from .gps import Prior
from .likelihoods import AbstractLikelihood, Gaussian
from .types import Dataset, PRNGKeyType
from .utils import concat_dictionaries


@dataclass
class AbstractVariationalFamily:
    """
    Abstract base class used to represent families of distributions that can be
    used within variational inference.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> dx.Distribution:
        """For a given set of parameters, compute the latent function's prediction
        under the variational approximation.

        Args:
            *args (Any): Arguments of the variational family's `predict` method.
            **kwargs (Any): Keyword arguments of the variational family's `predict`
                method.

        Returns:
            Any: The output of the variational family's `predict` method.
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """
        The parameters of the distribution. For example, the multivariate
        Gaussian would return a mean vector and covariance matrix.

        Args:
            key (PRNGKeyType): The PRNG key used to initialise the parameters.

        Returns:
            Dict: The parameters of the distribution.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> dx.Distribution:
        """Predict the GP's output given the input.

        Args:
            *args (Any): Arguments of the variational family's ``predict``
            method.
            **kwargs (Any): Keyword arguments of the variational family's
            ``predict`` method.

        Returns:
            Any: The output of the variational family's ``predict`` method.
        """
        raise NotImplementedError


@dataclass
class AbstractVariationalGaussian(AbstractVariationalFamily):
    """The variational Gaussian family of probability distributions."""

    prior: Prior
    inducing_inputs: Float[Array, "N D"]
    name: str = "Gaussian"

    def __post_init__(self):
        """Initialise the variational Gaussian distribution."""
        self.num_inducing = self.inducing_inputs.shape[0]


@dataclass
class VariationalGaussian(AbstractVariationalGaussian):
    """The variational Gaussian family of probability distributions.

    The variational family is q(f(·)) = ∫ p(f(·)|u) q(u) du, where
    :math:`u = f(z)` are the function values at the inducing inputs z
    and the distribution over the inducing inputs is
    :math:`q(u) = \\mathcal{N}(\\mu, S)`.  We parameterise this over
    :math:`\\mu` and sqrt with S = sqrt sqrtᵀ.
    """

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """
        Return the variational mean vector, variational root covariance matrix,
        and inducing input vector that parameterise the variational Gaussian
        distribution.

        Args:
            key (PRNGKeyType): The PRNG key used to initialise the parameters.

        Returns:
            Dict: The parameters of the distribution.
        """
        m = self.num_inducing

        return concat_dictionaries(
            self.prior._initialise_params(key),
            {
                "variational_family": {
                    "inducing_inputs": self.inducing_inputs,
                    "moments": {
                        "variational_mean": jnp.zeros((m, 1)),
                        "variational_root_covariance": jnp.eye(m),
                    },
                }
            },
        )

    def prior_kl(self, params: Dict) -> Float[Array, "1"]:
        """
        Compute the KL-divergence between our variational approximation and the
        Gaussian process prior.

        For this variational family, we have KL[q(f(·))||p(·)] = KL[q(u)||p(u)]
        = KL[ N(μ, S) || N(μz, Kzz) ], where u = f(z) and z are the inducing
        inputs.

        Args:
            params (Dict): The parameters at which our variational distribution
                and GP prior are to be evaluated.

        Returns:
             Float[Array, "1"]: The KL-divergence between our variational
                approximation and the GP prior.
        """

        jitter = get_defaults()["jitter"]

        # Unpack variational parameters
        mu = params["variational_family"]["moments"]["variational_mean"]
        sqrt = params["variational_family"]["moments"]["variational_root_covariance"]
        z = params["variational_family"]["inducing_inputs"]
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel computation
        gram = kernel.gram

        μz = mean_function(params["mean_function"], z)
        Kzz = gram(kernel, params["kernel"], z)
        Kzz += I(m) * jitter
        Lz = Kzz.triangular_lower()

        qu = dx.MultivariateNormalTri(jnp.atleast_1d(mu.squeeze()), sqrt)
        pu = dx.MultivariateNormalTri(jnp.atleast_1d(μz.squeeze()), Lz)

        return kld_dense_dense(qu, pu)

    def predict(
        self, params: Dict
    ) -> Callable[[Float[Array, "N D"]], dx.MultivariateNormalTri]:
        """
        Compute the predictive distribution of the GP at the test inputs t.

        This is the integral q(f(t)) = ∫ p(f(t)|u) q(u) du, which can be
        computed in closed form as:

            N[f(t); μt + Ktz Kzz⁻¹ (μ - μz),  Ktt - Ktz Kzz⁻¹ Kzt + Ktz Kzz⁻¹ S Kzz⁻¹ Kzt ].

        Args:
            params (Dict): The set of parameters that are to be used to
                parameterise our variational approximation and GP.

        Returns:
            Callable[[Float[Array, "N D"]], dx.MultivariateNormalTri]: A
                function that accepts a set of test points and will return
                the predictive distribution at those points.
        """
        jitter = get_defaults()["jitter"]

        # Unpack variational parameters
        mu = params["variational_family"]["moments"]["variational_mean"]
        sqrt = params["variational_family"]["moments"]["variational_root_covariance"]
        z = params["variational_family"]["inducing_inputs"]
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel computation
        gram = kernel.gram
        cross_covariance = kernel.cross_covariance

        Kzz = gram(kernel, params["kernel"], z)
        Kzz += I(m) * jitter
        Lz = Kzz.triangular_lower()
        μz = mean_function(params["mean_function"], z)

        def predict_fn(
            test_inputs: Float[Array, "N D"]
        ) -> dx.MultivariateNormalFullCovariance:

            # Unpack test inputs
            t, n_test = test_inputs, test_inputs.shape[0]

            Ktt = gram(kernel, params["kernel"], t)
            Kzt = cross_covariance(kernel, params["kernel"], z, t)
            μt = mean_function(params["mean_function"], t)

            # Lz⁻¹ Kzt
            Lz_inv_Kzt = jsp.linalg.solve_triangular(Lz, Kzt, lower=True)

            # Kzz⁻¹ Kzt
            Kzz_inv_Kzt = jsp.linalg.solve_triangular(Lz.T, Lz_inv_Kzt, lower=False)

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
            covariance += I(n_test) * jitter

            return dx.MultivariateNormalFullCovariance(
                jnp.atleast_1d(mean.squeeze()), covariance.to_dense()
            )

        return predict_fn


@dataclass
class WhitenedVariationalGaussian(VariationalGaussian):
    """
    The whitened variational Gaussian family of probability distributions.

    The variational family is q(f(·)) = ∫ p(f(·)|u) q(u) du, where u = f(z)
    are the function values at the inducing inputs z and the distribution over
    the inducing inputs is q(u) = N(Lz μ + mz, Lz S Lzᵀ). We parameterise this
    over μ and sqrt with S = sqrt sqrtᵀ.

    """

    name: str = "Whitened variational Gaussian"

    def prior_kl(self, params: Dict) -> Float[Array, "1"]:
        """Compute the KL-divergence between our variational approximation and
        the Gaussian process prior.

        For this variational family, we have KL[q(f(·))||p(·)] = KL[q(u)||p(u)] = KL[N(μ, S)||N(0, I)].

        Args:
            params (Dict): The parameters at which our variational distribution
                and GP prior are to be evaluated.

        Returns:
            Float[Array, "1"]: The KL-divergence between our variational
                approximation and the GP prior.
        """

        # Unpack variational parameters
        mu = params["variational_family"]["moments"]["variational_mean"]
        sqrt = params["variational_family"]["moments"]["variational_root_covariance"]

        # Compute whitened KL divergence
        qu = dx.MultivariateNormalTri(jnp.atleast_1d(mu.squeeze()), sqrt)
        return kld_dense_white(qu)

    def predict(
        self, params: Dict
    ) -> Callable[[Float[Array, "N D"]], dx.MultivariateNormalFullCovariance]:
        """Compute the predictive distribution of the GP at the test inputs t.

        This is the integral q(f(t)) = ∫ p(f(t)|u) q(u) du, which can be computed in closed form as

            N[f(t); μt  +  Ktz Lz⁻ᵀ μ,  Ktt  -  Ktz Kzz⁻¹ Kzt  +  Ktz Lz⁻ᵀ S Lz⁻¹ Kzt].

        Args:
            params (Dict): The set of parameters that are to be used to parameterise our variational approximation and GP.

        Returns:
            Callable[[Float[Array, "N D"]], dx.MultivariateNormalTri]: A function that accepts a set of test points and will return the predictive distribution at those points.
        """
        jitter = get_defaults()["jitter"]

        # Unpack variational parameters
        mu = params["variational_family"]["moments"]["variational_mean"]
        sqrt = params["variational_family"]["moments"]["variational_root_covariance"]
        z = params["variational_family"]["inducing_inputs"]
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel computation
        gram = kernel.gram
        cross_covariance = kernel.cross_covariance

        Kzz = gram(kernel, params["kernel"], z)
        Kzz += I(m) * jitter
        Lz = Kzz.triangular_lower()

        def predict_fn(
            test_inputs: Float[Array, "N D"]
        ) -> dx.MultivariateNormalFullCovariance:

            # Unpack test inputs
            t, n_test = test_inputs, test_inputs.shape[0]

            Ktt = gram(kernel, params["kernel"], t)
            Kzt = cross_covariance(kernel, params["kernel"], z, t)
            μt = mean_function(params["mean_function"], t)

            # Lz⁻¹ Kzt
            Lz_inv_Kzt = jsp.linalg.solve_triangular(Lz, Kzt, lower=True)

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
            covariance += I(n_test) * jitter

            return dx.MultivariateNormalFullCovariance(
                jnp.atleast_1d(mean.squeeze()), covariance.to_dense()
            )

        return predict_fn


@dataclass
class NaturalVariationalGaussian(AbstractVariationalGaussian):
    """The natural variational Gaussian family of probability distributions.

    The variational family is q(f(·)) = ∫ p(f(·)|u) q(u) du, where u = f(z) are the function values at the inducing inputs z
    and the distribution over the inducing inputs is q(u) = N(μ, S). Expressing the variational distribution, in the form of the
    exponential family, q(u) = exp(θᵀ T(u) - a(θ)), gives rise to the natural paramerisation θ = (θ₁, θ₂) = (S⁻¹μ, -S⁻¹/2), to perform
    model inference, where T(u) = [u, uuᵀ] are the sufficient statistics.

    """

    name: str = "Natural Gaussian"

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Return the natural vector and matrix, inducing inputs, and hyperparameters that parameterise the natural Gaussian distribution."""

        m = self.num_inducing

        return concat_dictionaries(
            self.prior._initialise_params(key),
            {
                "variational_family": {
                    "inducing_inputs": self.inducing_inputs,
                    "moments": {
                        "natural_vector": jnp.zeros((m, 1)),
                        "natural_matrix": -0.5 * jnp.eye(m),
                    },
                }
            },
        )

    def prior_kl(self, params: Dict) -> Float[Array, "1"]:
        """Compute the KL-divergence between our current variational approximation and the Gaussian process prior.

        For this variational family, we have KL[q(f(·))||p(·)] = KL[q(u)||p(u)] = KL[N(μ, S)||N(mz, Kzz)],

        with μ and S computed from the natural paramerisation θ = (S⁻¹μ, -S⁻¹/2).

        Args:
            params (Dict): The parameters at which our variational distribution and GP prior are to be evaluated.

        Returns:
            Float[Array, "1"]: The KL-divergence between our variational approximation and the GP prior.
        """
        jitter = get_defaults()["jitter"]

        # Unpack variational parameters
        natural_vector = params["variational_family"]["moments"]["natural_vector"]
        natural_matrix = params["variational_family"]["moments"]["natural_matrix"]
        z = params["variational_family"]["inducing_inputs"]
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel computation
        gram = kernel.gram

        # S⁻¹ = -2θ₂
        S_inv = -2 * natural_matrix
        S_inv += jnp.eye(m) * jitter

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

        μz = mean_function(params["mean_function"], z)
        Kzz = gram(kernel, params["kernel"], z)
        Kzz += I(m) * jitter
        Lz = Kzz.triangular_lower()

        qu = dx.MultivariateNormalTri(jnp.atleast_1d(mu.squeeze()), sqrt)
        pu = dx.MultivariateNormalTri(jnp.atleast_1d(μz.squeeze()), Lz)

        return kld_dense_dense(qu, pu)

    def predict(
        self, params: Dict
    ) -> Callable[[Float[Array, "N D"]], dx.MultivariateNormalFullCovariance]:
        """Compute the predictive distribution of the GP at the test inputs t.

        This is the integral q(f(t)) = ∫ p(f(t)|u) q(u) du, which can be computed in closed form as

             N[f(t); μt + Ktz Kzz⁻¹ (μ - μz),  Ktt - Ktz Kzz⁻¹ Kzt + Ktz Kzz⁻¹ S Kzz⁻¹ Kzt ],

        with μ and S computed from the natural paramerisation θ = (S⁻¹μ, -S⁻¹/2).

        Args:
            params (Dict): The set of parameters that are to be used to parameterise our variational approximation and GP.

        Returns:
            Callable[[Float[Array, "N D"]], dx.MultivariateNormalTri]: A function that accepts a set of test points and will return the predictive distribution at those points.
        """
        jitter = get_defaults()["jitter"]

        # Unpack variational parameters
        natural_vector = params["variational_family"]["moments"]["natural_vector"]
        natural_matrix = params["variational_family"]["moments"]["natural_matrix"]
        z = params["variational_family"]["inducing_inputs"]
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel computation
        gram = kernel.gram
        cross_covariance = kernel.cross_covariance

        # S⁻¹ = -2θ₂
        S_inv = -2 * natural_matrix
        S_inv += jnp.eye(m) * jitter

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

        Kzz = gram(kernel, params["kernel"], z)
        Kzz += I(m) * jitter
        Lz = Kzz.triangular_lower()
        μz = mean_function(params["mean_function"], z)

        def predict_fn(test_inputs: Float[Array, "N D"]) -> dx.MultivariateNormalTri:

            # Unpack test inputs
            t, n_test = test_inputs, test_inputs.shape[0]

            Ktt = gram(kernel, params["kernel"], t)
            Kzt = cross_covariance(kernel, params["kernel"], z, t)
            μt = mean_function(params["mean_function"], t)

            # Lz⁻¹ Kzt
            Lz_inv_Kzt = jsp.linalg.solve_triangular(Lz, Kzt, lower=True)

            # Kzz⁻¹ Kzt
            Kzz_inv_Kzt = jsp.linalg.solve_triangular(Lz.T, Lz_inv_Kzt, lower=False)

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
            covariance += I(n_test) * jitter

            return dx.MultivariateNormalFullCovariance(
                jnp.atleast_1d(mean.squeeze()), covariance.to_dense()
            )

        return predict_fn


@dataclass
class ExpectationVariationalGaussian(AbstractVariationalGaussian):
    """The natural variational Gaussian family of probability distributions.

    The variational family is q(f(·)) = ∫ p(f(·)|u) q(u) du, where u = f(z) are the function values at the inducing inputs z
    and the distribution over the inducing inputs is q(u) = N(μ, S). Expressing the variational distribution, in the form of the
    exponential family, q(u) = exp(θᵀ T(u) - a(θ)), gives rise to the natural paramerisation θ = (θ₁, θ₂) = (S⁻¹μ, -S⁻¹/2) and
    sufficient stastics T(u) = [u, uuᵀ]. The expectation parameters are given by η = ∫ T(u) q(u) du. This gives a parameterisation,
    η = (η₁, η₁) = (μ, S + uuᵀ) to perform model inference over.
    """

    name: str = "Expectation Gaussian"

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Return the expectation vector and matrix, inducing inputs, and hyperparameters that parameterise the expectation Gaussian distribution."""

        self.num_inducing = self.inducing_inputs.shape[0]

        m = self.num_inducing

        return concat_dictionaries(
            self.prior._initialise_params(key),
            {
                "variational_family": {
                    "inducing_inputs": self.inducing_inputs,
                    "moments": {
                        "expectation_vector": jnp.zeros((m, 1)),
                        "expectation_matrix": jnp.eye(m),
                    },
                }
            },
        )

    def prior_kl(self, params: Dict) -> Float[Array, "1"]:
        """Compute the KL-divergence between our current variational approximation and the Gaussian process prior.

        For this variational family, we have KL[q(f(·))||p(·)] = KL[q(u)||p(u)] = KL[N(μ, S)||N(mz, Kzz)],

        with μ and S computed from the expectation paramerisation η = (μ, S + uuᵀ).

        Args:
            params (Dict): The parameters at which our variational distribution and GP prior are to be evaluated.

        Returns:
            Float[Array, "1"]: The KL-divergence between our variational approximation and the GP prior.
        """
        jitter = get_defaults()["jitter"]

        # Unpack variational parameters
        expectation_vector = params["variational_family"]["moments"][
            "expectation_vector"
        ]
        expectation_matrix = params["variational_family"]["moments"][
            "expectation_matrix"
        ]
        z = params["variational_family"]["inducing_inputs"]
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel compuation
        gram = kernel.gram

        # μ = η₁
        mu = expectation_vector

        # S = η₂ - η₁ η₁ᵀ
        S = expectation_matrix - jnp.outer(mu, mu)
        S += jnp.eye(m) * jitter

        # S = sqrt sqrtᵀ
        sqrt = jnp.linalg.cholesky(S)

        μz = mean_function(params["mean_function"], z)
        Kzz = gram(kernel, params["kernel"], z)
        Kzz += I(m) * jitter
        Lz = Kzz.triangular_lower()

        qu = dx.MultivariateNormalTri(jnp.atleast_1d(mu.squeeze()), sqrt)
        pu = dx.MultivariateNormalTri(jnp.atleast_1d(μz.squeeze()), Lz)

        return kld_dense_dense(qu, pu)

    def predict(
        self, params: Dict
    ) -> Callable[[Float[Array, "N D"]], dx.MultivariateNormalFullCovariance]:
        """Compute the predictive distribution of the GP at the test inputs t.

        This is the integral q(f(t)) = ∫ p(f(t)|u) q(u) du, which can be computed in closed form as

             N[f(t); μt + Ktz Kzz⁻¹ (μ - μz),  Ktt - Ktz Kzz⁻¹ Kzt + Ktz Kzz⁻¹ S Kzz⁻¹ Kzt ],

        with μ and S computed from the expectation paramerisation η = (μ, S + uuᵀ).

        Args:
            params (Dict): The set of parameters that are to be used to parameterise our variational approximation and GP.

        Returns:
            Callable[[Float[Array, "N D"]], dx.MultivariateNormalTri]: A function that accepts a set of test points and will return the predictive distribution at those points.
        """
        jitter = get_defaults()["jitter"]

        # Unpack variational parameters
        expectation_vector = params["variational_family"]["moments"][
            "expectation_vector"
        ]
        expectation_matrix = params["variational_family"]["moments"][
            "expectation_matrix"
        ]
        z = params["variational_family"]["inducing_inputs"]
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel compuation
        gram = kernel.gram
        cross_covariance = kernel.cross_covariance

        # μ = η₁
        mu = expectation_vector

        # S = η₂ - η₁ η₁ᵀ
        S = expectation_matrix - jnp.matmul(mu, mu.T)
        S += jnp.eye(m) * jitter

        # S = sqrt sqrtᵀ
        sqrt = jnp.linalg.cholesky(S)

        Kzz = gram(kernel, params["kernel"], z)
        Kzz += I(m) * jitter
        Lz = Kzz.triangular_lower()
        μz = mean_function(params["mean_function"], z)

        def predict_fn(
            test_inputs: Float[Array, "N D"]
        ) -> dx.MultivariateNormalFullCovariance:

            # Unpack test inputs
            t, n_test = test_inputs, test_inputs.shape[0]

            Ktt = gram(kernel, params["kernel"], t)
            Kzt = cross_covariance(kernel, params["kernel"], z, t)
            μt = mean_function(params["mean_function"], t)

            # Lz⁻¹ Kzt
            Lz_inv_Kzt = jsp.linalg.solve_triangular(Lz, Kzt, lower=True)

            # Kzz⁻¹ Kzt
            Kzz_inv_Kzt = jsp.linalg.solve_triangular(Lz.T, Lz_inv_Kzt, lower=False)

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
            covariance += I(n_test) * jitter

            return dx.MultivariateNormalFullCovariance(
                jnp.atleast_1d(mean.squeeze()), covariance.to_dense()
            )

        return predict_fn


@dataclass
class CollapsedVariationalGaussian(AbstractVariationalFamily):
    """Collapsed variational Gaussian family of probability distributions.
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables in Sparse Gaussian Processes."""

    prior: Prior
    likelihood: AbstractLikelihood
    inducing_inputs: Float[Array, "M D"]
    name: str = "Collapsed variational Gaussian"
    diag: Optional[bool] = False

    def __post_init__(self):
        """Initialise the variational Gaussian distribution."""
        self.num_inducing = self.inducing_inputs.shape[0]

        if not isinstance(self.likelihood, Gaussian):
            raise TypeError("Likelihood must be Gaussian.")

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Return the variational mean vector, variational root covariance matrix, and inducing input vector that parameterise the variational Gaussian distribution."""
        return concat_dictionaries(
            self.prior._initialise_params(key),
            {
                "variational_family": {"inducing_inputs": self.inducing_inputs},
                "likelihood": {
                    "obs_noise": self.likelihood._initialise_params(key)["obs_noise"]
                },
            },
        )

    def predict(
        self,
        params: Dict,
        train_data: Dataset,
    ) -> Callable[[Float[Array, "N D"]], dx.MultivariateNormalFullCovariance]:
        """Compute the predictive distribution of the GP at the test inputs.

        Args:
            params (Dict): The set of parameters that are to be used to parameterise our variational approximation and GP.
            train_data (Dataset): The training data that was used to fit the GP.

        Returns:
            Callable[[Float[Array, "N D"]], dx.MultivariateNormalTri]: A function that accepts a set of test points and will return the predictive distribution at those points.
        """
        jitter = get_defaults()["jitter"]

        def predict_fn(
            test_inputs: Float[Array, "N D"]
        ) -> dx.MultivariateNormalFullCovariance:
            # TODO - can we cache some of this?

            # Unpack test inputs
            t, n_test = test_inputs, test_inputs.shape[0]

            # Unpack training data
            x, y = train_data.X, train_data.y

            # Unpack variational parameters
            noise = params["likelihood"]["obs_noise"]
            z = params["variational_family"]["inducing_inputs"]
            m = self.num_inducing

            # Unpack mean function and kernel
            mean_function = self.prior.mean_function
            kernel = self.prior.kernel

            # Unpack kernel compuation
            gram = kernel.gram
            cross_covariance = kernel.cross_covariance

            Kzx = cross_covariance(kernel, params["kernel"], z, x)
            Kzz = gram(kernel, params["kernel"], z)
            Kzz += I(m) * jitter

            # Lz Lzᵀ = Kzz
            Lz = Kzz.triangular_lower()

            # Lz⁻¹ Kzx
            Lz_inv_Kzx = jsp.linalg.solve_triangular(Lz, Kzx, lower=True)

            # A = Lz⁻¹ Kzt / σ
            A = Lz_inv_Kzx / jnp.sqrt(noise)

            # AAᵀ
            AAT = jnp.matmul(A, A.T)

            # LLᵀ = I + AAᵀ
            L = jnp.linalg.cholesky(jnp.eye(m) + AAT)

            μx = mean_function(params["mean_function"], x)
            diff = y - μx

            # Lz⁻¹ Kzx (y - μx)
            Lz_inv_Kzx_diff = jsp.linalg.cho_solve(
                (L, True), jnp.matmul(Lz_inv_Kzx, diff)
            )

            # Kzz⁻¹ Kzx (y - μx)
            Kzz_inv_Kzx_diff = jsp.linalg.solve_triangular(
                Lz.T, Lz_inv_Kzx_diff, lower=False
            )

            Ktt = gram(kernel, params["kernel"], t)
            Kzt = cross_covariance(kernel, params["kernel"], z, t)
            μt = mean_function(params["mean_function"], t)

            # Lz⁻¹ Kzt
            Lz_inv_Kzt = jsp.linalg.solve_triangular(Lz, Kzt, lower=True)

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
            covariance += I(n_test) * jitter

            return dx.MultivariateNormalFullCovariance(
                jnp.atleast_1d(mean.squeeze()), covariance.to_dense()
            )

        return predict_fn


# TODO: Abstract these out to a KL divergence that accepts a linear operator to facilate structured covarainces other than dense.
def kld_dense_dense(
    q: dx.MultivariateNormalTri, p: dx.MultivariateNormalTri
) -> Float[Array, "1"]:
    """Kullback-Leibler divergence KL[q(x)||p(x)] between two dense covariance Gaussian distributions
             q(x) = N(x; μq, Σq) and p(x) = N(x; μp, Σp).

    Args:
        q (dx.MultivariateNormalTri): A multivariate Gaussian distribution.
        p (dx.MultivariateNormalTri): A multivariate Gaussian distribution.

    Returns:
        Float[Array, "1"]: The KL divergence between the two distributions.
    """

    q_mu = q.loc
    q_sqrt = q.scale_tri
    n = q_mu.shape[-1]

    p_mu = p.loc
    p_sqrt = p.scale_tri

    diag = jnp.diag(q_sqrt)

    # Trace term tr(Σp⁻¹ Σq)
    trace = jnp.sum(jnp.square(jsp.linalg.solve_triangular(p_sqrt, q_sqrt, lower=True)))

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    alpha = jsp.linalg.solve_triangular(p_sqrt, p_mu - q_mu, lower=True)
    mahalanobis = jnp.sum(jnp.square(alpha))

    # log|Σq|
    logdet_qcov = jnp.sum(jnp.log(jnp.square(diag)))
    two_kl = mahalanobis - n - logdet_qcov + trace

    # log|Σp|
    log_det_pcov = jnp.sum(jnp.log(jnp.square(jnp.diag(p_sqrt))))
    two_kl += log_det_pcov

    return two_kl / 2.0


def kld_dense_white(q: dx.MultivariateNormalTri) -> Float[Array, "1"]:
    """Kullback-Leibler divergence KL[q(x)||p(x)] between a dense covariance Gaussian distribution
        q(x) = N(x; μq, Σq), and white indenity Gaussian p(x) = N(x; 0, I).

        This is useful for variational inference with a whitened variational family.

      Args:
        q (dx.MultivariateNormalTri): A multivariate Gaussian distribution.

    Returns:
        Float[Array, "1"]: The KL divergence between the two distributions.
    """

    q_mu = q.loc
    q_sqrt = q.scale_tri
    n = q_mu.shape[-1]

    diag = jnp.diag(q_sqrt)

    # Trace term tr(Σp⁻¹ Σq), and alpha for Mahalanobis term:
    alpha = q_mu
    trace = jnp.sum(jnp.square(q_sqrt))

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = jnp.sum(jnp.square(alpha))

    # log|Σq| (no log|Σp| as this is just zero!)
    logdet_qcov = jnp.sum(jnp.log(jnp.square(diag)))
    two_kl = mahalanobis - n - logdet_qcov + trace

    return two_kl / 2.0


__all__ = [
    "AbstractVariationalFamily",
    "AbstractVariationalGaussian",
    "VariationalGaussian",
    "WhitenedVariationalGaussian",
    "NaturalVariationalGaussian",
    "ExpectationVariationalGaussian",
    "CollapsedVariationalGaussian",
]
