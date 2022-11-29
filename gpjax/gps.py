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
from typing import Any, Callable, Dict, Optional

import distrax as dx
import jax.numpy as jnp
import jax.random as jr
from chex import dataclass
from jaxtyping import Array, Float

from jaxlinop import identity

from .config import get_defaults
from .kernels import AbstractKernel
from .likelihoods import AbstractLikelihood, Conjugate, Gaussian, NonConjugate
from .mean_functions import AbstractMeanFunction, Zero
from .types import Dataset, PRNGKeyType
from .utils import concat_dictionaries
from .gaussian_distribution import GaussianDistribution


@dataclass
class AbstractPrior:
    """Abstract Gaussian process prior.

    All Gaussian processes priors should inherit from this class.

    All GPJax Modules are `Chex dataclasses <https://docs.python.org/3/library/dataclasses.html>`_. Since
    dataclasses take over ``__init__``, the ``__post_init__`` method can be used to
    initialise the GP's parameters.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> dx.Distribution:
        """Evaluate the Gaussian process at the given points. The output of this function
        is a `Distrax distribution <https://github.com/deepmind/distrax>`_ from which the
        the latent function's mean and covariance can be evaluated and the distribution
        can be sampled.

        Under the hood, ``__call__`` is calling the objects ``predict`` method. For this
        reasons, classes inheriting the ``AbstractPrior`` class, should not overwrite the
        ``__call__`` method and should instead define a ``predict`` method.

        Args:
            *args (Any): The arguments to pass to the GP's `predict` method.
            **kwargs (Any): The keyword arguments to pass to the GP's `predict` method.

        Returns:
            dx.Distribution: A multivariate normal random variable representation of the Gaussian process.
        """
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> dx.Distribution:
        """Compute the latent function's multivariate normal distribution for a
        given set of parameters. For any class inheriting the ``AbstractPrior`` class,
        this method must be implemented.

        Args:
            *args (Any): Arguments to the predict method.
            **kwargs (Any): Keyword arguments to the predict method.

        Returns:
            dx.Distribution: A multivariate normal random variable representation of the Gaussian process.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """An initialisation method for the GP's parameters. This method should
        be implemented for all classes that inherit the ``AbstractPrior`` class.
        Whilst not always necessary, the method accepts a PRNG key to allow
        for stochastic initialisation. The method should is most often invoked
        through the ``initialise`` function given in GPJax.

        Args:
            key (PRNGKeyType): The PRNG key.

        Returns:
            Dict: The initialised parameter set.
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

    Attributes:
        kernel (AbstractKernel): The kernel function used to parameterise the prior.
        mean_function (MeanFunction): The mean function used to parameterise the
            prior. Defaults to zero.
        name (str): The name of the GP prior. Defaults to "GP prior".
    """

    kernel: AbstractKernel
    mean_function: Optional[AbstractMeanFunction] = Zero()
    name: Optional[str] = "GP prior"

    def __mul__(self, other: AbstractLikelihood):
        """The product of a prior and likelihood is proportional to the
        posterior distribution. By computing the product of a GP prior and a
        likelihood object, a posterior GP object will be returned. Mathetically,
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

    def predict(
        self, params: Dict
    ) -> Callable[[Float[Array, "N D"]], GaussianDistribution]:
        """Compute the predictive prior distribution for a given set of
        parameters. The output of this function is a function that computes
        a distrx distribution for a given set of inputs.

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
            params (Dict): The specific set of parameters for which the mean
            function should be defined for.

        Returns:
            Callable[[Float[Array, "N D"]], GaussianDistribution]: A mean
            function that accepts an input array for where the mean function
            should be evaluated at. The mean function's value at these points is
            then returned.
        """
        jitter = get_defaults()["jitter"]

        # Unpack mean function and kernel
        mean_function = self.mean_function
        kernel = self.kernel

        # Unpack kernel computation
        gram = kernel.gram

        def predict_fn(test_inputs: Float[Array, "N D"]) -> GaussianDistribution:

            # Unpack test inputs
            t = test_inputs
            n_test = test_inputs.shape[0]

            μt = mean_function(params["mean_function"], t)
            Ktt = gram(kernel, params["kernel"], t)
            Ktt += identity(n_test) * jitter

            return GaussianDistribution(jnp.atleast_1d(μt.squeeze()), Ktt)

        return predict_fn

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Initialise the GP prior's parameter set.

        Args:
            key (PRNGKeyType): The PRNG key.

        Returns:
            Dict: The initialised parameter set.
        """
        return {
            "kernel": self.kernel._initialise_params(key),
            "mean_function": self.mean_function._initialise_params(key),
        }


#######################
# GP Posteriors
#######################
@dataclass
class AbstractPosterior(AbstractPrior):
    """The base GP posterior object conditioned on an observed dataset. All
    posterior objects should inherit from this class.

    All GPJax Modules are `Chex dataclasses
    <https://docs.python.org/3/library/dataclasses.html>`_. Since dataclasses
    take over ``__init__``, the ``__post_init__`` method can be used to
    initialise the GP's parameters.

    Attributes:
        prior (Prior): The prior distribution of the GP.
        likelihood (AbstractLikelihood): The likelihood distribution of the
            observed dataset.
        name (str): The name of the GP posterior. Defaults to "GP posterior".
    """

    prior: Prior
    likelihood: AbstractLikelihood
    name: Optional[str] = "GP posterior"

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> GaussianDistribution:
        """Compute the predictive posterior distribution of the latent function
        for a given set of parameters. For any class inheriting the
        ``AbstractPosterior`` class, this method must be implemented.

        Args:
            *args (Any): Arguments to the predict method. **kwargs (Any):
            Keyword arguments to the predict method.

        Returns:
            GaussianDistribution: A multivariate normal random variable
            representation of the Gaussian process.
        """
        raise NotImplementedError

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Initialise the parameter set of a GP posterior.

        Args:
            key (PRNGKeyType): The PRNG key.

        Returns:
            Dict: The initialised parameter set.
        """
        return concat_dictionaries(
            self.prior._initialise_params(key),
            {"likelihood": self.likelihood._initialise_params(key)},
        )


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

    Attributes:
        prior (Prior): The prior distribution of the GP.
        likelihood (Gaussian): The Gaussian likelihood distribution of the observed dataset.
        name (str): The name of the GP posterior. Defaults to "Conjugate posterior".
    """

    prior: Prior
    likelihood: Gaussian
    name: Optional[str] = "Conjugate posterior"

    def predict(
        self,
        params: Dict,
        train_data: Dataset,
    ) -> Callable[[Float[Array, "N D"]], GaussianDistribution]:
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

        £xample:
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
            Callable[[Float[Array, "N D"]], GaussianDistribution]: A
                function that accepts an input array and returns the predictive
                distribution as a ``GaussianDistribution``.
        """
        jitter = get_defaults()["jitter"]

        # Unpack training data
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel computation
        gram = kernel.gram
        cross_covariance = kernel.cross_covariance

        # Observation noise σ²
        obs_noise = params["likelihood"]["obs_noise"]
        μx = mean_function(params["mean_function"], x)

        # Precompute Gram matrix, Kxx, at training inputs, x
        Kxx = gram(kernel, params["kernel"], x)
        Kxx += identity(n) * jitter

        # Σ = Kxx + Iσ²
        Sigma = Kxx + identity(n) * obs_noise

        def predict(test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
            """Compute the predictive distribution at a set of test inputs.

            Args:
                test_inputs (Float[Array, "N D"]): A Jax array of test inputs.

            Returns:
                GaussianDistribution: A ``GaussianDistribution``
                object that represents the predictive distribution.
            """

            # Unpack test inputs
            t = test_inputs
            n_test = test_inputs.shape[0]

            μt = mean_function(params["mean_function"], t)
            Ktt = gram(kernel, params["kernel"], t)
            Kxt = cross_covariance(kernel, params["kernel"], x, t)

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
        train_data: Dataset,
        negative: bool = False,
    ) -> Callable[[Dict], Float[Array, "1"]]:
        """Compute the marginal log-likelihood function of the Gaussian process.
        The returned function can then be used for gradient based optimisation
        of the model's parameters or for model comparison. The implementation
        given here enables exact estimation of the Gaussian process' latent
        function values.

        For a training dataset :math:`\\{x_n, y_n\\}_{n=1}^N`, set of test
        inputs :math:`\\mathbf{x}^{\\star}` the corresponding latent function
        evaluations are given by :math:`\\mathbf{f} = f(\\mathbf{x})`
        and :math:`\\mathbf{f}^{\\star} = f(\\mathbf{x}^{\\star})`, the marginal
        log-likelihood is given by:

        .. math::

            \\log p(\\mathbf{y}) & = \\int p(\\mathbf{y}\\mid\\mathbf{f})p(\\mathbf{f}, \\mathbf{f}^{\\star}\\mathrm{d}\\mathbf{f}^{\\star}\\\\
            &=0.5\\left(-\\mathbf{y}^{\\top}\\left(k(\\mathbf{x}, \\mathbf{x}') +\\sigma^2\\mathbf{I}_N  \\right)^{-1}\\mathbf{y}-\\log\\lvert k(\\mathbf{x}, \\mathbf{x}') + \\sigma^2\\mathbf{I}_N\\rvert - n\\log 2\\pi \\right).

        Example:

        For a given ``ConjugatePosterior`` object, the following code snippet shows
        how the marginal log-likelihood can be evaluated.

        >>> import gpjax as gpx
        >>>
        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)
        >>>
        >>> params = gpx.initialise(posterior)
        >>> mll = posterior.marginal_log_likelihood(train_data = D)
        >>> mll(params)

        Our goal is to maximise the marginal log-likelihood. Therefore, when
        optimising the model's parameters with respect to the parameters, we
        use the negative marginal log-likelihood. This can be realised through

        >>> mll = posterior.marginal_log_likelihood(train_data = D, negative=True)

        Further, prior distributions can be passed into the marginal log-likelihood

        >>> mll = posterior.marginal_log_likelihood(train_data = D)

        For optimal performance, the marginal log-likelihood should be ``jax.jit``
        compiled.

        >>> mll = jit(posterior.marginal_log_likelihood(train_data = D))

        Args:
            train_data (Dataset): The training dataset used to compute the
                marginal log-likelihood.
            negative (bool, optional): Whether or not the returned function
                should be negative. For optimisation, the negative is useful
                as minimisation of the negative marginal log-likelihood is
                equivalent to maximisation of the marginal log-likelihood.
                Defaults to False.

        Returns:
            Callable[[Dict], Float[Array, "1"]]: A functional representation
                of the marginal log-likelihood that can be evaluated at a
                given parameter set.
        """
        jitter = get_defaults()["jitter"]

        # Unpack training data
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel computation
        gram = kernel.gram

        # The sign of the marginal log-likelihood depends on whether we are maximising or minimising
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def mll(
            params: Dict,
        ):
            """Compute the marginal log-likelihood of the Gaussian process.

            Args:
                params (Dict): The model's parameters.

            Returns:
                Float[Array, "1"]: The marginal log-likelihood.
            """

            # Observation noise σ²
            obs_noise = params["likelihood"]["obs_noise"]
            μx = mean_function(params["mean_function"], x)

            # TODO: This implementation does not take advantage of the covariance operator structure.
            # Future work concerns implementation of a custom Gaussian distribution / measure object that accepts a covariance operator.

            # Σ = (Kxx + Iσ²) = LLᵀ
            Kxx = gram(kernel, params["kernel"], x)
            Kxx += identity(n) * jitter
            Sigma = Kxx + identity(n) * obs_noise

            # p(y | x, θ), where θ are the model hyperparameters:
            marginal_likelihood = GaussianDistribution(
                jnp.atleast_1d(μx.squeeze()), Sigma
            )

            return constant * (
                marginal_likelihood.log_prob(jnp.atleast_1d(y.squeeze())).squeeze()
            )

        return mll


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

    Attributes:
        prior (AbstractPrior): The Gaussian process prior distribution.
        likelihood (AbstractLikelihood): The likelihood function that
            represents the data.
        name (str): The name of the posterior object. Defaults to
            "Non-conjugate posterior".
    """

    prior: AbstractPrior
    likelihood: AbstractLikelihood
    name: Optional[str] = "Non-conjugate posterior"

    def _initialise_params(self, key: PRNGKeyType) -> Dict:
        """Initialise the parameter set of a non-conjugate GP posterior.

        Args:
            key (PRNGKeyType): A PRNG key used to initialise the parameters.

        Returns:
            Dict: A dictionary containing the default parameter set.
        """
        parameters = concat_dictionaries(
            self.prior._initialise_params(key),
            {"likelihood": self.likelihood._initialise_params(key)},
        )
        parameters["latent"] = jnp.zeros(shape=(self.likelihood.num_datapoints, 1))
        return parameters

    def predict(
        self,
        params: Dict,
        train_data: Dataset,
    ) -> Callable[[Float[Array, "N D"]], dx.Distribution]:
        """
        Conditional on a set of training data, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned
        function can be evaluated at a set of test inputs to compute the
        corresponding predictive density. Note, to gain predictions on the scale
        of the original data, the returned distribution will need to be
        transformed through the likelihood function's inverse link function.

        Args:
            params (Dict): A dictionary of parameters that should be used to
                compute the posterior.
            train_data (Dataset): A `gpx.Dataset` object that contains the input
                and output data used for training dataset.

        Returns:
            tp.Callable[[Array], dx.Distribution]: A function that accepts an
                input array and returns the predictive distribution as
                a ``dx.Distribution``.
        """
        jitter = get_defaults()["jitter"]

        # Unpack training data
        x, n = train_data.X, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel computation
        gram = kernel.gram
        cross_covariance = kernel.cross_covariance

        # Precompute lower triangular of Gram matrix, Lx, at training inputs, x
        Kxx = gram(kernel, params["kernel"], x)
        Kxx += identity(n) * jitter
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
            Ktx = cross_covariance(kernel, params["kernel"], t, x)
            Ktt = gram(kernel, params["kernel"], t) + identity(n_test) * jitter
            μt = mean_function(params["mean_function"], t)

            # Lx⁻¹ Kxt
            Lx_inv_Kxt = Lx.solve(Ktx.T)

            # Whitened function values, wx, correponding to the inputs, x
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
        NonConjugatePosterior object does not provide an exact marginal
        log-likelihood function. Instead, the NonConjugatePosterior object
        represents the posterior distributions as a function of the model's
        hyperparameters and the latent function. Markov chain Monte Carlo,
        variational inference, or Laplace approximations can then be used to
        sample from, or optimise an approximation to, the posterior
        distribution.

        Args:
            train_data (Dataset): The training dataset used to compute the
                marginal log-likelihood.
            negative (bool, optional): Whether or not the returned function
                should be negative. For optimisation, the negative is useful as
                minimisation of the negative marginal log-likelihood is equivalent
                to maximisation of the marginal log-likelihood. Defaults to False.

        Returns:
            Callable[[Dict], Float[Array, "1"]]: A functional representation
                of the marginal log-likelihood that can be evaluated at a given
                parameter set.
        """
        jitter = get_defaults()["jitter"]

        # Unpack dataset
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Unpack kernel computation
        gram = kernel.gram

        # Link function of the likelihood
        link_function = self.likelihood.link_function

        # The sign of the marginal log-likelihood depends on whether we are maximising or minimising
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def mll(params: Dict):
            """Compute the marginal log-likelihood of the model.

            Args:
                params (Dict): A dictionary of parameters that should be used
                    to compute the marginal log-likelihood.

            Returns:
                Float[Array, "1"]: The marginal log-likelihood of the model.
            """

            # Compute lower triangular of the kernel Gram matrix
            Kxx = gram(kernel, params["kernel"], x)
            Kxx += identity(n) * jitter
            Lx = Kxx.to_root()

            # Compute the prior mean function
            μx = mean_function(params["mean_function"], x)

            # Whitened function values, wx, correponding to the inputs, x
            wx = params["latent"]

            # f(x) = μx  +  Lx wx
            fx = μx + Lx @ wx

            # p(y | f(x), θ), where θ are the model hyperparameters
            likelihood = link_function(params, fx)

            # Whitened latent function values prior, p(wx | θ) = N(0, I)
            latent_prior = dx.Normal(loc=0.0, scale=1.0)

            return constant * (likelihood.log_prob(y).sum() + latent_prior.log_prob(wx).sum())

        return mll


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
    if isinstance(likelihood, Conjugate):
        PosteriorGP = ConjugatePosterior

    elif isinstance(likelihood, NonConjugate):
        PosteriorGP = NonConjugatePosterior

    else:
        raise NotImplementedError(
            f"No posterior implemented for {likelihood.name} likelihood"
        )

    return PosteriorGP(prior=prior, likelihood=likelihood)


__all__ = [
    "AbstractPrior",
    "Prior",
    "AbstractPosterior",
    "ConjugatePosterior",
    "NonConjugatePosterior",
    "construct_posterior",
]
