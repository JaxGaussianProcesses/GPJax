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

import deprecation
import distrax as dx
import jax.numpy as jnp
from jax.random import KeyArray
from jaxkern.base import AbstractKernel
from jaxlinop import identity
from jaxtyping import Array, Float
from jaxutils import Dataset, Identity, Parameters, PyTree

from .gaussian_distribution import GaussianDistribution
from .likelihoods import AbstractLikelihood, Conjugate, NonConjugate
from .mean_functions import AbstractMeanFunction, Zero
from .objectives import (
    AbstractObjective,
    ConjugateMarginalLogLikelihood,
    NonConjugateMarginalLogLikelihood,
)


class AbstractPrior(PyTree):
    """Abstract Gaussian process prior.

    All Gaussian processes priors should inherit from this class."""

    def __init__(self) -> None:
        super().__init__()
        self._jitter = 1e-6

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
    def init_params(self, key: KeyArray) -> Parameters:
        """An initialisation method for the GP's parameters. This method should
        be implemented for all classes that inherit the ``AbstractPrior`` class.
        Whilst not always necessary, the method accepts a PRNG key to allow
        for stochastic initialisation. The method should is most often invoked
        through the ``initialise`` function given in GPJax.

        Args:
            key (KeyArray): The PRNG key.

        Returns:
            Dict: The initialised parameter set.
        """
        raise NotImplementedError

    @deprecation.deprecated(
        deprecated_in="0.5.7",
        removed_in="0.6.0",
        details="Use the ``init_params`` method for parameter initialisation.",
    )
    def _initialise_params(self, key: KeyArray) -> Dict:
        """Deprecated method for initialising the GP's parameters. Succeeded by ``init_params``."""
        return self.init_params(key)

    @property
    def jitter(self) -> float:
        return self._jitter

    @jitter.setter
    def jitter(self, value: float):
        self._jitter = value


#######################
# GP Priors
#######################
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

    def __init__(
        self,
        kernel: AbstractKernel,
        mean_function: Optional[AbstractMeanFunction] = Zero(),
        name: Optional[str] = "GP prior",
    ) -> None:
        """Initialise the GP prior.

        Args:
            kernel (AbstractKernel): The kernel function used to parameterise the prior.
            mean_function (Optional[MeanFunction]): The mean function used to parameterise the
                prior. Defaults to zero.
            name (Optional[str]): The name of the GP prior. Defaults to "GP prior".
        """
        super().__init__()
        self.kernel = kernel
        self.mean_function = mean_function
        self.name = name

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

    def predict(
        self, params: Dict
    ) -> Callable[[Float[Array, "N D"]], GaussianDistribution]:
        """Compute the predictive prior distribution for a given set of
        parameters. The output of this function is a function that computes
        a Distrax distribution for a given set of inputs.

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

        # Unpack mean function and kernel
        mean_function = self.mean_function
        kernel = self.kernel

        def predict_fn(test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
            # Unpack test inputs
            t = test_inputs
            n_test = test_inputs.shape[0]

            μt = mean_function(params["mean_function"], t)
            Ktt = kernel.gram(params["kernel"], t)
            Ktt += identity(n_test) * self.jitter

            return GaussianDistribution(jnp.atleast_1d(μt.squeeze()), Ktt)

        return predict_fn

    def init_params(self, key: KeyArray) -> Parameters:
        """Initialise the GP prior's parameter set.

        Args:
            key (KeyArray): The PRNG key.

        Returns:
            Dict: The initialised parameter set.
        """
        kernel = self.kernel.init_params(key)
        meanf = self.mean_function.init_params(key)
        params = kernel.combine(meanf, left_key="kernel", right_key="mean_function")
        return params


#######################
# GP Posteriors
#######################
class AbstractPosterior(AbstractPrior):
    """The base GP posterior object conditioned on an observed dataset. All
    posterior objects should inherit from this class."""

    def __init__(
        self,
        prior: AbstractPrior,
        likelihood: AbstractLikelihood,
        name: Optional[str] = "GP posterior",
    ) -> None:
        """Initialise the GP posterior object.

        Args:
            prior (Prior): The prior distribution of the GP.
            likelihood (AbstractLikelihood): The likelihood distribution of the observed dataset.
            name (Optional[str]): The name of the GP posterior. Defaults to "GP posterior".
        """
        super().__init__()
        self.prior = prior
        self.likelihood = likelihood
        self.name = name

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

    def loss_function(self) -> AbstractObjective:
        raise NotImplementedError(f"No loss function defined for {self.name}.")

    def init_params(self, key: KeyArray) -> Parameters:
        """Initialise the parameter set of a GP posterior.

        Args:
            key (KeyArray): The PRNG key.

        Returns:
            Dict: The initialised parameter set.
        """
        prior_params = self.prior.init_params(key)
        likelihood_params = self.likelihood.init_params(key)
        prior_params.add_parameter(
            key="likelihood",
            parameter=likelihood_params,
        )
        return prior_params


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

    def __init__(
        self,
        prior: AbstractPrior,
        likelihood: AbstractLikelihood,
        name: Optional[str] = "GP posterior",
    ) -> None:
        """Initialise the conjugate GP posterior object.

        Args:
            prior (Prior): The prior distribution of the GP.
            likelihood (AbstractLikelihood): The likelihood distribution of the observed dataset.
            name (Optional[str]): The name of the GP posterior. Defaults to "GP posterior".
        """
        super().__init__(prior=prior, likelihood=likelihood)
        self.name = name

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
            Callable[[Float[Array, "N D"]], GaussianDistribution]: A
                function that accepts an input array and returns the predictive
                distribution as a ``GaussianDistribution``.
        """
        # Unpack training data
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Observation noise σ²
        obs_noise = params["likelihood"]["obs_noise"]
        μx = mean_function(params["mean_function"], x)

        # Precompute Gram matrix, Kxx, at training inputs, x
        Kxx = kernel.gram(params["kernel"], x)
        Kxx += identity(n) * self.jitter

        # Σ = Kxx + Iσ²
        Sigma = Kxx + identity(n) * obs_noise

        def predict(test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
            """Compute the predictive distribution at a set of test inputs.

            Args:
                test_inputs (Float[Array, "N D"]): A Jax array of test inputs.

            Returns:
                A ``GaussianDistribution`` object that represents the
                predictive distribution.
            """

            # Unpack test inputs
            t = test_inputs
            n_test = test_inputs.shape[0]

            μt = mean_function(params["mean_function"], t)
            Ktt = kernel.gram(params["kernel"], t)
            Kxt = kernel.cross_covariance(params["kernel"], x, t)

            # Σ⁻¹ Kxt
            Sigma_inv_Kxt = Sigma.solve(Kxt)

            # μt  +  Ktx (Kxx + Iσ²)⁻¹ (y  -  μx)
            mean = μt + jnp.matmul(Sigma_inv_Kxt.T, y - μx)

            # Ktt  -  Ktx (Kxx + Iσ²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
            covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
            covariance += identity(n_test) * self.jitter

            return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)

        return predict

    def loss_function(self) -> ConjugateMarginalLogLikelihood:
        return ConjugateMarginalLogLikelihood(model=self, negative=True)


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

    def __init__(
        self,
        prior: AbstractPrior,
        likelihood: AbstractLikelihood,
        name: Optional[str] = "GP posterior",
    ) -> None:
        """Initialise a non-conjugate Gaussian process posterior object.

        Args:
            prior (AbstractPrior): The Gaussian process prior distribution.
            likelihood (AbstractLikelihood): The likelihood function that represents the data.
            name (Optional[str]): The name of the posterior object. Defaults to "GP posterior".
        """
        super().__init__(prior=prior, likelihood=likelihood)
        self.name = name

    def init_params(self, key: KeyArray) -> Parameters:
        """Initialise the parameter set of a non-conjugate GP posterior.

        Args:
            key (KeyArray): A PRNG key used to initialise the parameters.

        Returns:
            DParametersict: A `Parameters` object containing the default parameter set.
        """
        params = self.prior.init_params(key)
        likelihood_params = self.likelihood.init_params(key)
        params.add_parameter(key="likelihood", parameter=likelihood_params)
        params.add_parameter(
            "latent",
            value=jnp.zeros(shape=(self.likelihood.num_datapoints, 1)),
            prior=dx.Normal(0.0, 1.0),
            trainability=True,
            bijector=Identity,
        )
        return params

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
            Callable[[Array], dx.Distribution]: A function that accepts an
                input array and returns the predictive distribution as
                a ``dx.Distribution``.
        """
        # Unpack training data
        x, n = train_data.X, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Precompute lower triangular of Gram matrix, Lx, at training inputs, x
        Kxx = kernel.gram(params["kernel"], x)
        Kxx += identity(n) * self.jitter
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
            Ktt = kernel.gram(params["kernel"], t) + identity(n_test) * self.jitter
            μt = mean_function(params["mean_function"], t)

            # Lx⁻¹ Kxt
            Lx_inv_Kxt = Lx.solve(Ktx.T)

            # Whitened function values, wx, corresponding to the inputs, x
            wx = params["latent"]

            # μt + Ktx Lx⁻¹ wx
            mean = μt + jnp.matmul(Lx_inv_Kxt.T, wx)

            # Ktt - Ktx Kxx⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
            covariance = Ktt - jnp.matmul(Lx_inv_Kxt.T, Lx_inv_Kxt)
            covariance += identity(n_test) * self.jitter

            return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)

        return predict_fn

    def loss_function(self) -> NonConjugateMarginalLogLikelihood:
        return NonConjugateMarginalLogLikelihood(model=self, negative=True)


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
