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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict

import deprecation
import distrax as dx
import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array, Float
from jaxutils import Dataset, PyTree

from .config import get_global_config
from .kernels import AbstractKernel
from .likelihoods import AbstractLikelihood
from .mean_functions import AbstractMeanFunction, Zero
from jaxutils import Dataset
from .utils import concat_dictionaries
from .gaussian_distribution import GaussianDistribution
from .kernels import AbstractKernel
from .kernels.base import AbstractKernel
from .likelihoods import AbstractLikelihood, Conjugate, NonConjugate
from .linops import identity
from .mean_functions import AbstractMeanFunction, Zero
from .utils import concat_dictionaries


@dataclass
class AbstractPrior(Module):
    """Abstract Gaussian process prior."""

    kernel: AbstractKernel
    mean_function: AbstractMeanFunction
    jitter: float = static_field(1e-6)

    def __call__(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        """Evaluate the Gaussian process at the given points. The output of this function
        is a `TensorFlow probability distribution <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions>`_ from which the
        the latent function's mean and covariance can be evaluated and the distribution
        can be sampled.

        Under the hood, ``__call__`` is calling the objects ``predict`` method. For this
        reasons, classes inheriting the ``AbstractPrior`` class, should not overwrite the
        ``__call__`` method and should instead define a ``predict`` method.

        Args:
            *args (Any): The arguments to pass to the GP's `predict` method.
            **kwargs (Any): The keyword arguments to pass to the GP's `predict` method.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation of the Gaussian process.
        """
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        """Compute the latent function's multivariate normal distribution for a
        given set of parameters. For any class inheriting the ``AbstractPrior`` class,
        this method must be implemented.

        Args:
            *args (Any): Arguments to the predict method.
            **kwargs (Any): Keyword arguments to the predict method.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation of the Gaussian process.
        """
        raise NotImplementedError


#######################
# GP Priors
#######################
@dataclass
class Prior(AbstractPrior):
    """A Gaussian process prior object. The GP is parameterised by a
    `mean <https://gpjax.readthedocs.io/en/latest/api.html#module-gpjax.mean_functions>`_
    and `kernel <https://gpjax.readthedocs.io/en/latest/api.html#module-gpjax.kernels>`_ function.

    A Gaussian process prior parameterised by a mean function :math:`m(\\cdot)` and a kernel
    function :math:`k(\\cdot, \\cdot)` is given by

    .. math::

        p(f(\\cdot)) = \mathcal{GP}(m(\\cdot), k(\\cdot, \\cdot)).

    To invoke a ``Prior`` distribution, only a kernel function is required. By
    default, the mean function will be set to zero. In general, this assumption
    will be reasonable assuming the data being modelled has been centred.

    Example:
        >>> import gpjax as gpx
        >>>
        >>> kernel = gpx.kernels.RBF()
        >>> prior = gpx.Prior(kernel = kernel)
    """

    def __mul__(self, other: AbstractLikelihood):
        """The product of a prior and likelihood is proportional to the
        posterior distribution. By computing the product of a GP prior and a
        likelihood object, a posterior GP object will be returned. Mathematically,
        this can be described by:
         .. math::

             p(f(\\cdot) | y) \\propto p(y | f(\\cdot)) p(f(\\cdot)).

         where :math:`p(y | f(\\cdot))` is the likelihood and :math:`p(f(\\cdot))`
         is the prior.


         Example:
             >>> import gpjax as gpx
             >>>
             >>> kernel = gpx.kernels.RBF()
             >>> prior = gpx.Prior(kernel = kernel)
             >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=100)
             >>>
             >>> prior * likelihood

         Args:
             other (Likelihood): The likelihood distribution of the observed dataset.

         Returns:
             Posterior: The relevant GP posterior for the given prior and
                likelihood. Special cases are accounted for where the model
                is conjugate.
        """
        return construct_posterior(prior=self, likelihood=other)

    def __rmul__(self, other: AbstractLikelihood):
        """Reimplement the multiplication operator to allow for order-invariant
        product of a likelihood and a prior i.e., likelihood * prior.

        Args:
            other (Likelihood): The likelihood distribution of the observed
                dataset.

        Returns:
            Posterior: The relevant GP posterior for the given prior and
                likelihood. Special cases are accounted for where the model
                is conjugate.
        """
        return self.__mul__(other)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        """Compute the predictive prior distribution for a given set of
        parameters. The output of this function is a function that computes
        a TFP distribution for a given set of inputs.

        In the following example, we compute the predictive prior distribution
        and then evaluate it on the interval :math:`[0, 1]`:

        Example:
            >>> import gpjax as gpx
            >>> import jax.numpy as jnp
            >>>
            >>> kernel = gpx.kernels.RBF()
            >>> prior = gpx.Prior(kernel = kernel)
            >>>
            >>> parameter_state = gpx.initialise(prior)
            >>> prior_predictive = prior.predict(parameter_state.params)
            >>> prior_predictive(jnp.linspace(0, 1, 100))

        Args:
            test_inputs (Float[Array, "N D"]): The inputs at which to evaluate the prior distribution.

        Returns:
            GaussianDistribution: A mean
            function that accepts an input array for where the mean function
            should be evaluated at. The mean function's value at these points is
            then returned.
        """
        x = test_inputs
        mx = self.mean_function(x)
        Kxx = self.kernel.gram(x)
        Kxx += identity(x.shape[0]) * self.jitter

        return GaussianDistribution(jnp.atleast_1d(mx.squeeze()), Kxx)

    def sample_approx(
        self,
        num_samples: int,
        key: KeyArray,
        num_features: Optional[int] = 100,
    ) -> FunctionalSample:
        r"""Build an approximate sample from the Gaussian process prior. This method
        provides a function that returns the evaluations of a sample across any given
        inputs.

        In particular, we approximate the Gaussian processes' prior as the finite feature
        approximation

        .. math:: \hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i

        def predict_fn(test_inputs: Float[Array, "N D"]) -> GaussianDistribution:

        where :math:`\phi_i` are m features sampled from the Fourier feature decomposition of
        the model's kernel and :math:`\theta_i` are samples from a unit Gaussian.


        A key property of such functional samples is that the same sample draw is
        evaluated for all queries. Consistency is a property that is prohibitively costly
        to ensure when sampling exactly from the GP prior, as the cost of exact sampling
        scales cubically with the size of the sample. In contrast, finite feature representations
        can be evaluated with constant cost regardless of the required number of queries.

        In the following example, we build 10 such samples
        and then evaluate them over the interval :math:`[0, 1]`:

        Example:
            For a ``prior`` distribution, the following code snippet will
            build and evaluate an approximate sample.

            >>> import gpjax as gpx
            >>> import jax.numpy as jnp
            >>>
            >>> sample_fn = prior.sample_appox(10, key)
            >>> sample_fn(jnp.linspace(0, 1, 100))

        Args:
            num_samples (int): The desired number of samples.
            params (Dict): The specific set of parameters for which the sample
            should be generated for.
            key (KeyArray): The random seed used for the sample(s).
            num_features (int): The number of features used when approximating the
            kernel.


        Returns:
            FunctionalSample: A function representing an approximate sample from the Gaussian
            process prior.
        """
        if (not isinstance(num_features, int)) or num_features <= 0:
            raise ValueError(f"num_features must be a positive integer")
        if (not isinstance(num_samples, int)) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer")

        approximate_kernel = RFF(base_kernel=self.kernel, num_basis_fns=num_features)
        feature_weights = normal(key, [num_samples, 2 * num_features])  # [B, L]

        def sample_fn(test_inputs: Float[Array, "N D"]) -> Float[Array, "N B"]:

            feature_evals = approximate_kernel.compute_features(x=test_inputs)
            feature_evals *= jnp.sqrt(self.kernel.variance / num_features)
            evaluated_sample = jnp.inner(feature_evals, feature_weights)  # [N, B]
            return self.mean_function(test_inputs) + evaluated_sample

        return sample_fn


#######################
# GP Posteriors
#######################
@dataclass
class AbstractPosterior(Module):
    """The base GP posterior object conditioned on an observed dataset. All
    posterior objects should inherit from this class."""

    prior: AbstractPrior
    likelihood: AbstractLikelihood
    jitter: float = static_field(1e-6)

    def __call__(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        """Evaluate the Gaussian process at the given points. The output of this function
        is a `TFP distribution <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions>`_ from which the
        the latent function's mean and covariance can be evaluated and the distribution
        can be sampled.

        Under the hood, ``__call__`` is calling the objects ``predict`` method. For this
        reasons, classes inheriting the ``AbstractPrior`` class, should not overwrite the
        ``__call__`` method and should instead define a ``predict`` method.

        Args:
            *args (Any): The arguments to pass to the GP's `predict` method.
            **kwargs (Any): The keyword arguments to pass to the GP's `predict` method.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation of the Gaussian process.
        """
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        """Compute the latent function's multivariate normal distribution for a
        given set of parameters. For any class inheriting the ``AbstractPrior`` class,
        this method must be implemented.

        Args:
            *args (Any): Arguments to the predict method.
            **kwargs (Any): Keyword arguments to the predict method.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation of the Gaussian process.
        """
        raise NotImplementedError


@dataclass
class ConjugatePosterior(AbstractPosterior):
    """A Gaussian process posterior distribution when the constituent likelihood
    function is a Gaussian distribution. In such cases, the latent function values
    :math:`f` can be analytically integrated out of the posterior distribution.
    As such, many computational operations can be simplified; something we make use
    of in this object.

    For a Gaussian process prior :math:`p(\mathbf{f})` and a Gaussian likelihood
    :math:`p(y | \\mathbf{f}) = \\mathcal{N}(y\\mid \mathbf{f}, \\sigma^2))` where
    :math:`\mathbf{f} = f(\\mathbf{x})`, the predictive posterior distribution at
    a set of inputs :math:`\\mathbf{x}` is given by

    .. math::

        p(\\mathbf{f}^{\\star}\mid \mathbf{y}) & = \\int p(\\mathbf{f}^{\\star} \\mathbf{f} \\mid \\mathbf{y})\\\\
        & =\\mathcal{N}(\\mathbf{f}^{\\star} \\boldsymbol{\mu}_{\mid \mathbf{y}}, \\boldsymbol{\Sigma}_{\mid \mathbf{y}}
    where

    .. math::

        \\boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\\mathbf{x}^{\\star}, \\mathbf{x})\\left(k(\\mathbf{x}, \\mathbf{x}')+\\sigma^2\\mathbf{I}_n\\right)^{-1}\\mathbf{y}  \\\\
        \\boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\\mathbf{x}^{\\star}, \\mathbf{x}^{\\star\\prime}) -k(\\mathbf{x}^{\\star}, \\mathbf{x})\\left( k(\\mathbf{x}, \\mathbf{x}') + \\sigma^2\\mathbf{I}_n \\right)^{-1}k(\\mathbf{x}, \\mathbf{x}^{\\star}).

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp
        >>>
        >>> prior = gpx.Prior(kernel = gpx.kernels.RBF())
        >>> likelihood = gpx.likelihoods.Gaussian()
        >>>
        >>> posterior = prior * likelihood
    """

    def predict(
        self,
        test_inputs: Float[Array, "N D"],
        train_data: Dataset,
    ) -> GaussianDistribution:
        """Conditional on a training data set, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned
        function can be evaluated at a set of test inputs to compute the
        corresponding predictive density.

        The predictive distribution of a conjugate GP is given by
        .. math::

            p(\\mathbf{f}^{\\star}\mid \mathbf{y}) & = \\int p(\\mathbf{f}^{\\star} \\mathbf{f} \\mid \\mathbf{y})\\\\
            & =\\mathcal{N}(\\mathbf{f}^{\\star} \\boldsymbol{\mu}_{\mid \mathbf{y}}, \\boldsymbol{\Sigma}_{\mid \mathbf{y}}
        where

        .. math::

            \\boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\\mathbf{x}^{\\star}, \\mathbf{x})\\left(k(\\mathbf{x}, \\mathbf{x}')+\\sigma^2\\mathbf{I}_n\\right)^{-1}\\mathbf{y}  \\\\
            \\boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\\mathbf{x}^{\\star}, \\mathbf{x}^{\\star\\prime}) -k(\\mathbf{x}^{\\star}, \\mathbf{x})\\left( k(\\mathbf{x}, \\mathbf{x}') + \\sigma^2\\mathbf{I}_n \\right)^{-1}k(\\mathbf{x}, \\mathbf{x}^{\\star}).

        The conditioning set is a GPJax ``Dataset`` object, whilst predictions
        are made on a regular Jax array.

        Example:
            For a ``posterior`` distribution, the following code snippet will
            evaluate the predictive distribution.

            >>> import gpjax as gpx
            >>>
            >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
            >>> ytrain = jnp.sin(xtrain)
            >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
            >>>
            >>> params = gpx.initialise(posterior)
            >>> predictive_dist = posterior.predict(params, gpx.Dataset(X=xtrain, y=ytrain))
            >>> predictive_dist(xtest)

        Args:
            params (Dict): A dictionary of parameters that should be used to
                compute the posterior.
            train_data (Dataset): A `gpx.Dataset` object that contains the
                input and output data used for training dataset.

        Returns:
            GaussianDistribution: A
                function that accepts an input array and returns the predictive
                distribution as a ``GaussianDistribution``.
        """
        # Unpack training data
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack test inputs
        t, n_test = test_inputs, test_inputs.shape[0]

        # Observation noise σ²
        obs_noise = self.likelihood.obs_noise
        mx = self.prior.mean_function(x)

        # Precompute Gram matrix, Kxx, at training inputs, x
        Kxx = self.prior.kernel.gram(x) + (identity(n) * self.prior.jitter)

        # Σ = Kxx + Iσ²
        Sigma = Kxx + identity(n) * obs_noise

        μt = self.prior.mean_function(t)
        Ktt = self.prior.kernel.gram(t)
        Kxt = self.prior.kernel.cross_covariance(x, t)

        # Σ⁻¹ Kxt
        Sigma_inv_Kxt = Sigma.solve(Kxt)

        # μt  +  Ktx (Kxx + Iσ²)⁻¹ (y  -  μx)
        mean = μt + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

        # Ktt  -  Ktx (Kxx + Iσ²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += identity(n_test) * self.prior.jitter

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)

            # Σ⁻¹ Kxt
            Sigma_inv_Kxt = Sigma.solve(Kxt)

            # μt  +  Ktx (Kxx + Iσ²)⁻¹ (y  -  μx)
            mean = μt + jnp.matmul(Sigma_inv_Kxt.T, y - μx)

            # Ktt  -  Ktx (Kxx + Iσ²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
            covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
            covariance += identity(n_test) * jitter

            return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)

        return predict

    def marginal_log_likelihood(
        self,
        num_samples: int,
        train_data: Dataset,
        key: KeyArray,
        num_features: Optional[int] = 100,
    ) -> FunctionalSample:
        r"""Build an approximate sample from the Gaussian process posterior. This method
        provides a function that returns the evaluations of a sample across any given
        inputs.

        Unlike when building approximate samples from a Gaussian process prior, decompositions
        based on Fourier features alone rarely give accurate samples. Therefore, we must also
        include an additional set of features (known as canonical features) to better model the
        transition from Gaussian process prior to Gaussian process posterior. For more details
        see https://arxiv.org/pdf/2002.09309.pdf

        In particular, we approximate the Gaussian processes' posterior as the finite feature
        approximation

        .. math:: \hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i + \sum{j=1}^N v_jk(.,x_j)


        where :math:`\phi_i` are m features sampled from the Fourier feature decomposition of
        the model's kernel and :math:`k(., x_j)` are N canonical features. The Fourier
        weights :math:`\theta_i` are samples from a unit Gaussian.
        See https://arxiv.org/pdf/2002.09309.pdf for expressions for the canonical
        weights :math:`v_j`.


        A key property of such functional samples is that the same sample draw is
        evaluated for all queries. Consistency is a property that is prohibitively costly
        to ensure when sampling exactly from the GP prior, as the cost of exact sampling
        scales cubically with the size of the sample. In contrast, finite feature representations
        can be evaluated with constant cost regardless of the required number of queries.

        Args:
            num_samples (int): The desired number of samples.
            key (KeyArray): The random seed used for the sample(s).
            num_features (int): The number of features used when approximating the
            kernel.


        Returns:
            FunctionalSample: A function representing an approximate sample from the Gaussian
            process prior.
        """
        if (not isinstance(num_features, int)) or num_features <= 0:
            raise ValueError(f"num_features must be a positive integer")
        if (not isinstance(num_samples, int)) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer")

        # Approximate kernel with feature decomposition
        approximate_kernel = RFF(
            base_kernel=self.prior.kernel, num_basis_fns=num_features
        )

        def eval_fourier_features(
            test_inputs: Float[Array, "N D"]
        ) -> Float[Array, "N L"]:
            Phi = approximate_kernel.compute_features(x=test_inputs)
            Phi *= jnp.sqrt(self.prior.kernel.variance / num_features)
            return Phi

        # sample weights for Fourier features
        fourier_weights = normal(key, [num_samples, 2 * num_features])  # [B, L]

        # sample weights v for canonical features
        # v = Σ⁻¹ (y + ε - ɸ⍵) for  Σ = Kxx + Iσ² and ε ᯈ N(0, σ²)
        Kxx = self.prior.kernel.gram(train_data.X)  #  [N, N]
        Sigma = Kxx + identity(train_data.n) * (
            self.likelihood.obs_noise + self.jitter
        )  #  [N, N]
        eps = jnp.sqrt(self.likelihood.obs_noise) * normal(
            key, [train_data.n, num_samples]
        )  #  [N, B]
        y = train_data.y - self.prior.mean_function(train_data.X)  # account for mean
        Phi = eval_fourier_features(train_data.X)
        canonical_weights = Sigma.solve(
            y + eps - jnp.inner(Phi, fourier_weights)
        )  #  [N, B]

        def sample_fn(test_inputs: Float[Array, "n D"]) -> Float[Array, "n B"]:
            fourier_features = eval_fourier_features(test_inputs)
            weight_space_contribution = jnp.inner(
                fourier_features, fourier_weights
            )  # [n, B]
            canonical_features = self.prior.kernel.cross_covariance(
                test_inputs, train_data.X
            )  # [n, N]
            function_space_contribution = jnp.matmul(
                canonical_features, canonical_weights
            )

            return constant * (
                marginal_likelihood.log_prob(jnp.atleast_1d(y.squeeze())).squeeze()
            )

        return sample_fn


@dataclass
class NonConjugatePosterior(AbstractPosterior):
    """
    A Gaussian process posterior object for models where the likelihood is
    non-Gaussian. Unlike the ``ConjugatePosterior`` object, the
    ``NonConjugatePosterior`` object does not provide an exact marginal
    log-likelihood function. Instead, the ``NonConjugatePosterior`` object
    represents the posterior distributions as a function of the model's
    hyperparameters and the latent function. Markov chain Monte Carlo,
    variational inference, or Laplace approximations can then be used to sample
    from, or optimise an approximation to, the posterior distribution.
    """

    latent: Float[Array, "N 1"] = param_field(None)
    key: KeyArray = static_field(PRNGKey(42))

    def init_params(self, key: KeyArray) -> Dict:
        """Initialise the parameter set of a non-conjugate GP posterior.

        Args:
            key (KeyArray): A PRNG key used to initialise the parameters.

        Returns:
            Dict: A dictionary containing the default parameter set.
        """
        parameters = concat_dictionaries(
            self.prior.init_params(key),
            {"likelihood": self.likelihood.init_params(key)},
        )
        parameters["latent"] = jnp.zeros(shape=(self.likelihood.num_datapoints, 1))
        return parameters

    def predict(
        self, test_inputs: Float[Array, "N D"], train_data: Dataset
    ) -> GaussianDistribution:
        """
        Conditional on a set of training data, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned
        function can be evaluated at a set of test inputs to compute the
        corresponding predictive density. Note, to gain predictions on the scale
        of the original data, the returned distribution will need to be
        transformed through the likelihood function's inverse link function.

        Args:
            train_data (Dataset): A `gpx.Dataset` object that contains the input
                and output data used for training dataset.

        Returns:
            GaussianDistribution: A function that accepts an
                input array and returns the predictive distribution as
                a ``dx.Distribution``.
        """
        # Unpack training data
        x, n = train_data.X, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Precompute lower triangular of Gram matrix, Lx, at training inputs, x
        Kxx = kernel.gram(x)
        Kxx += identity(n) * self.prior.jitter
        Lx = Kxx.to_root()

        def predict_fn(test_inputs: Float[Array, "N D"]) -> dx.Distribution:
            """Predictive distribution of the latent function for a given set of test inputs.

            Args:
                test_inputs (Float[Array, "N D"]): A set of test inputs.

            Returns:
                dx.Distribution: The predictive distribution of the latent function.
            """

            # Unpack test inputs
            t, n_test = test_inputs, test_inputs.shape[0]

            # Compute terms of the posterior predictive distribution
            Ktx = kernel.cross_covariance(params["kernel"], t, x)
            Ktt = kernel.gram(params["kernel"], t) + identity(n_test) * jitter
            μt = mean_function(params["mean_function"], t)

            # Lx⁻¹ Kxt
            Lx_inv_Kxt = Lx.solve(Ktx.T)

            # Whitened function values, wx, corresponding to the inputs, x
            wx = params["latent"]

            # μt + Ktx Lx⁻¹ wx
            mean = μt + jnp.matmul(Lx_inv_Kxt.T, wx)

            # Ktt - Ktx Kxx⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
            covariance = Ktt - jnp.matmul(Lx_inv_Kxt.T, Lx_inv_Kxt)
            covariance += identity(n_test) * jitter

            return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)

        return predict_fn

    def marginal_log_likelihood(
        self,
        train_data: Dataset,
        negative: bool = False,
    ) -> Callable[[Dict], Float[Array, "1"]]:
        """
        Compute the marginal log-likelihood function of the Gaussian process.
        The returned function can then be used for gradient based optimisation
        of the model's parameters or for model comparison. The implementation
        given here is general and will work for any likelihood support by GPJax.

        Unlike the marginal_log_likelihood function of the ConjugatePosterior
        object, the marginal_log_likelihood function of the
        ``NonConjugatePosterior`` object does not provide an exact marginal
        log-likelihood function. Instead, the ``NonConjugatePosterior`` object
        represents the posterior distributions as a function of the model's
        hyperparameters and the latent function. Markov chain Monte Carlo,
        variational inference, or Laplace approximations can then be used to
        sample from, or optimise an approximation to, the posterior
        distribution.

        Args:
            train_data (Dataset): The training dataset used to compute the
                marginal log-likelihood.
            negative (Optional[bool]): Whether or not the returned function
                should be negative. For optimisation, the negative is useful as
                minimisation of the negative marginal log-likelihood is equivalent
                to maximisation of the marginal log-likelihood. Defaults to False.

        Returns:
            Callable[[Dict], Float[Array, "1"]]: A functional representation
                of the marginal log-likelihood that can be evaluated at a given
                parameter set.
        """
        jitter = get_global_config()["jitter"]

        # Unpack dataset
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Link function of the likelihood
        link_function = self.likelihood.link_function

        # The sign of the marginal log-likelihood depends on whether we are maximising or minimising
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        # Compute terms of the posterior predictive distribution
        Ktx = kernel.cross_covariance(t, x)
        Ktt = kernel.gram(t) + identity(n_test) * self.prior.jitter
        μt = mean_function(t)

        # Lx⁻¹ Kxt
        Lx_inv_Kxt = Lx.solve(Ktx.T)

        # Whitened function values, wx, corresponding to the inputs, x
        wx = self.latent

        # μt + Ktx Lx⁻¹ wx
        mean = μt + jnp.matmul(Lx_inv_Kxt.T, wx)

        # Ktt - Ktx Kxx⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Lx_inv_Kxt.T, Lx_inv_Kxt)
        covariance += identity(n_test) * self.prior.jitter

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)


def construct_posterior(
    prior: Prior, likelihood: AbstractLikelihood
) -> AbstractPosterior:
    """Utility function for constructing a posterior object from a prior and
    likelihood. The function will automatically select the correct posterior
    object based on the likelihood.

    Args:
        prior (Prior): The Prior distribution.
        likelihood (AbstractLikelihood): The likelihood that represents our
            beliefs around the distribution of the data.

    Returns:
        AbstractPosterior: A posterior distribution. If the likelihood is
            Gaussian, then a ``ConjugatePosterior`` will be returned. Otherwise,
            a ``NonConjugatePosterior`` will be returned.
    """
    if isinstance(likelihood, Gaussian):
        return ConjugatePosterior(prior=prior, likelihood=likelihood)

    return NonConjugatePosterior(prior=prior, likelihood=likelihood)


__all__ = [
    "AbstractPrior",
    "Prior",
    "AbstractPosterior",
    "ConjugatePosterior",
    "NonConjugatePosterior",
    "construct_posterior",
]
