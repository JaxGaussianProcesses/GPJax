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
# from __future__ import annotations
from abc import abstractmethod

import beartype.typing as tp
from cola.annotations import PSD
from cola.linalg.algorithm_base import Algorithm
from cola.linalg.decompositions.decompositions import Cholesky
from cola.linalg.inverse.inv import solve
from cola.ops.operators import I_like
from flax import nnx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Float,
    Num,
)

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.kernels import RFF
from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import (
    AbstractLikelihood,
    Gaussian,
    NonGaussian,
)
from gpjax.lower_cholesky import lower_cholesky
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.parameters import (
    Parameter,
    Real,
    Static,
)
from gpjax.typing import (
    Array,
    FunctionalSample,
    KeyArray,
)

K = tp.TypeVar("K", bound=AbstractKernel)
M = tp.TypeVar("M", bound=AbstractMeanFunction)
L = tp.TypeVar("L", bound=AbstractLikelihood)
NGL = tp.TypeVar("NGL", bound=NonGaussian)
GL = tp.TypeVar("GL", bound=Gaussian)


class AbstractPrior(nnx.Module, tp.Generic[M, K]):
    r"""Abstract Gaussian process prior."""

    def __init__(
        self,
        kernel: K,
        mean_function: M,
        jitter: float = 1e-6,
    ):
        r"""Construct a Gaussian process prior.

        Args:
            kernel: kernel object inheriting from AbstractKernel.
            mean_function: mean function object inheriting from AbstractMeanFunction.
        """
        self.kernel = kernel
        self.mean_function = mean_function
        self.jitter = jitter

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> GaussianDistribution:
        r"""Evaluate the Gaussian process at the given points.

        The output of this function is a
        [TensorFlow probability distribution](https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions) from which the
        the latent function's mean and covariance can be evaluated and the distribution
        can be sampled.

        Under the hood, `__call__` is calling the objects `predict` method. For this
        reasons, classes inheriting the `AbstractPrior` class, should not overwrite the
        `__call__` method and should instead define a `predict` method.

        Args:
            *args (Any): The arguments to pass to the GP's `predict` method.
            **kwargs (Any): The keyword arguments to pass to the GP's `predict` method.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation
                of the Gaussian process.
        """
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(self, *args: tp.Any, **kwargs: tp.Any) -> GaussianDistribution:
        r"""Evaluate the predictive distribution.

        Compute the latent function's multivariate normal distribution for a
        given set of parameters. For any class inheriting the `AbstractPrior` class,
        this method must be implemented.

        Args:
            *args (Any): Arguments to the predict method.
            **kwargs (Any): Keyword arguments to the predict method.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation
                of the Gaussian process.
        """
        raise NotImplementedError


#######################
# GP Priors
#######################
class Prior(AbstractPrior[M, K]):
    r"""A Gaussian process prior object.

    The GP is parameterised by a
    [mean](https://docs.jaxgaussianprocesses.com/api/mean_functions/)
    and [kernel](https://docs.jaxgaussianprocesses.com/api/kernels/base/)
    function.

    A Gaussian process prior parameterised by a mean function $m(\cdot)$ and a kernel
    function $k(\cdot, \cdot)$ is given by
    $p(f(\cdot)) = \mathcal{GP}(m(\cdot), k(\cdot, \cdot))$.

    To invoke a `Prior` distribution, a kernel and mean function must be specified.

    Example:
    ```python
        >>> import gpjax as gpx
        >>> kernel = gpx.kernels.RBF()
        >>> meanf = gpx.mean_functions.Zero()
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel = kernel)
    ```
    """

    if tp.TYPE_CHECKING:

        @tp.overload
        def __mul__(self, other: GL) -> "ConjugatePosterior[Prior[M, K], GL]": ...

        @tp.overload
        def __mul__(  # noqa: F811
            self, other: NGL
        ) -> "NonConjugatePosterior[Prior[M, K], NGL]": ...

        @tp.overload
        def __mul__(  # noqa: F811
            self, other: L
        ) -> "AbstractPosterior[Prior[M, K], L]": ...

    def __mul__(self, other):  # noqa: F811
        r"""Combine the prior with a likelihood to form a posterior distribution.

        The product of a prior and likelihood is proportional to the posterior
        distribution. By computing the product of a GP prior and a likelihood
        object, a posterior GP object will be returned. Mathematically, this can
        be described by:
        ```math
        p(f(\cdot) \mid y) \propto p(y \mid f(\cdot))p(f(\cdot)),
        ```
        where $p(y | f(\cdot))$ is the likelihood and $p(f(\cdot))$ is the prior.

        Example:
        ```pycon
            >>> import gpjax as gpx
            >>> meanf = gpx.mean_functions.Zero()
            >>> kernel = gpx.kernels.RBF()
            >>> prior = gpx.gps.Prior(mean_function=meanf, kernel = kernel)
            >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=100)
            >>> prior * likelihood
        ```
        Args:
            other (Likelihood): The likelihood distribution of the observed dataset.

        Returns
            Posterior: The relevant GP posterior for the given prior and
                likelihood. Special cases are accounted for where the model
                is conjugate.
        """
        return construct_posterior(prior=self, likelihood=other)

    if tp.TYPE_CHECKING:

        @tp.overload
        def __rmul__(self, other: GL) -> "ConjugatePosterior[Prior[M, K], GL]": ...

        @tp.overload
        def __rmul__(  # noqa: F811
            self, other: NGL
        ) -> "NonConjugatePosterior[Prior[M, K], NGL]": ...

        @tp.overload
        def __rmul__(  # noqa: F811
            self, other: L
        ) -> "AbstractPosterior[Prior[M, K], L]": ...

    def __rmul__(self, other):  # noqa: F811
        r"""Combine the prior with a likelihood to form a posterior distribution.

        Reimplement the multiplication operator to allow for order-invariant
        product of a likelihood and a prior i.e., likelihood * prior.

        Args:
            other (Likelihood): The likelihood distribution of the observed
                dataset.

        Returns
            Posterior: The relevant GP posterior for the given prior and
                likelihood. Special cases are accounted for where the model
                is conjugate.
        """
        return self.__mul__(other)

    def predict(self, test_inputs: Num[Array, "N D"]) -> GaussianDistribution:
        r"""Compute the predictive prior distribution for a given set of
        parameters. The output of this function is a function that computes
        a TFP distribution for a given set of inputs.

        In the following example, we compute the predictive prior distribution
        and then evaluate it on the interval :math:`[0, 1]`:

        Example:
        ```pycon
            >>> import gpjax as gpx
            >>> import jax.numpy as jnp
            >>> kernel = gpx.kernels.RBF()
            >>> mean_function = gpx.mean_functions.Zero()
            >>> prior = gpx.gps.Prior(mean_function=mean_function, kernel=kernel)
            >>> prior.predict(jnp.linspace(0, 1, 100)[:, None])
        ```

        Args:
            test_inputs (Float[Array, "N D"]): The inputs at which to evaluate the
                prior distribution.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation
                of the Gaussian process.
        """
        x = test_inputs
        mx = self.mean_function(x)
        Kxx = self.kernel.gram(x)
        Kxx += I_like(Kxx) * self.jitter
        Kxx = PSD(Kxx)

        return GaussianDistribution(jnp.atleast_1d(mx.squeeze()), Kxx)

    def sample_approx(
        self,
        num_samples: int,
        key: KeyArray,
        num_features: tp.Optional[int] = 100,
    ) -> FunctionalSample:
        r"""Approximate samples from the Gaussian process prior.

        Build an approximate sample from the Gaussian process prior. This method
        provides a function that returns the evaluations of a sample across any
        given inputs.

        In particular, we approximate the Gaussian processes' prior as the
        finite feature approximation
        $\hat{f}(x) = \sum_{i=1}^m\phi_i(x)\theta_i$ where $\phi_i$ are $m$ features
        sampled from the Fourier feature decomposition of the model's kernel and
        $\theta_i$ are samples from a unit Gaussian.

        A key property of such functional samples is that the same sample draw is
        evaluated for all queries. Consistency is a property that is prohibitively costly
        to ensure when sampling exactly from the GP prior, as the cost of exact sampling
        scales cubically with the size of the sample. In contrast, finite feature representations
        can be evaluated with constant cost regardless of the required number of queries.

        In the following example, we build 10 such samples and then evaluate them
        over the interval $[0, 1]$:

        For a `prior` distribution, the following code snippet will
        build and evaluate an approximate sample.

        Example:
        ```pycon
            >>> import gpjax as gpx
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> key = jr.PRNGKey(123)
            >>>
            >>> meanf = gpx.mean_functions.Zero()
            >>> kernel = gpx.kernels.RBF(n_dims=1)
            >>> prior = gpx.gps.Prior(mean_function=meanf, kernel = kernel)
            >>>
            >>> sample_fn = prior.sample_approx(10, key)
            >>> sample_fn(jnp.linspace(0, 1, 100).reshape(-1, 1))
        ```

        Args:
            num_samples (int): The desired number of samples.
            key (KeyArray): The random seed used for the sample(s).
            num_features (int): The number of features used when approximating the
                kernel.

        Returns:
            FunctionalSample: A function representing an approximate sample from the
                Gaussian process prior.
        """

        if (not isinstance(num_samples, int)) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        # sample fourier features
        fourier_feature_fn = _build_fourier_features_fn(self, num_features, key)

        # sample fourier weights
        feature_weights = jr.normal(key, [num_samples, 2 * num_features])  # [B, L]

        def sample_fn(test_inputs: Float[Array, "N D"]) -> Float[Array, "N B"]:
            feature_evals = fourier_feature_fn(test_inputs)  # [N, L]
            evaluated_sample = jnp.inner(feature_evals, feature_weights)  # [N, B]
            return self.mean_function(test_inputs) + evaluated_sample

        return sample_fn


P = tp.TypeVar("P", bound=AbstractPrior)


#######################
# GP Posteriors
#######################
class AbstractPosterior(nnx.Module, tp.Generic[P, L]):
    r"""Abstract Gaussian process posterior.

    The base GP posterior object conditioned on an observed dataset. All
    posterior objects should inherit from this class.
    """

    def __init__(
        self,
        prior: AbstractPrior[M, K],
        likelihood: L,
        jitter: float = 1e-6,
    ):
        r"""Construct a Gaussian process posterior.

        Args:
            prior (AbstractPrior): The prior distribution.
            likelihood (AbstractLikelihood): The likelihood distribution.
            jitter (float): A small constant added to the diagonal of the
                covariance matrix to ensure numerical stability.
        """
        self.prior = prior
        self.likelihood = likelihood
        self.jitter = jitter

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> GaussianDistribution:
        r"""Evaluate the Gaussian process posterior at the given points.

        The output of this function is a
        [TFP distribution](https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/distributions)
        from which the the latent function's mean and covariance can be
        evaluated and the distribution can be sampled.

        Under the hood, `__call__` is calling the objects `predict` method. For this
        reasons, classes inheriting the `AbstractPrior` class, should not overwrite the
        `__call__` method and should instead define a `predict` method.

        Args:
            *args (Any): The arguments to pass to the GP's `predict` method.
            **kwargs (Any): The keyword arguments to pass to the GP's `predict` method.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation
                of the Gaussian process.
        """
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(self, *args: tp.Any, **kwargs: tp.Any) -> GaussianDistribution:
        r"""Compute the latent function's multivariate normal distribution for a
        given set of parameters. For any class inheriting the `AbstractPrior` class,
        this method must be implemented.

        Args:
            *args (Any): Arguments to the predict method.
            **kwargs (Any): Keyword arguments to the predict method.

        Returns:
            GaussianDistribution: A multivariate normal random variable representation
                of the Gaussian process.
        """
        raise NotImplementedError


class ConjugatePosterior(AbstractPosterior[P, GL]):
    r"""A Conjuate Gaussian process posterior object.

    A Gaussian process posterior distribution when the constituent likelihood
    function is a Gaussian distribution. In such cases, the latent function values
    $f$ can be analytically integrated out of the posterior distribution.
    As such, many computational operations can be simplified; something we make use
    of in this object.

    For a Gaussian process prior $p(\mathbf{f})$ and a Gaussian likelihood
    $p(y | \mathbf{f}) = \mathcal{N}(y\mid \mathbf{f}, \sigma^2))$ where
    $\mathbf{f} = f(\mathbf{x})$, the predictive posterior distribution at
    a set of inputs $\mathbf{x}$ is given by
    ```math
    \begin{align}
    p(\mathbf{f}^{\star}\mid \mathbf{y}) & = \int p(\mathbf{f}^{\star}, \mathbf{f} \mid \mathbf{y})\\
        & =\mathcal{N}(\mathbf{f}^{\star} \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}
    \end{align}
    ```
    where
    ```math
    \begin{align}
    \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y}  \\
    \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
    \end{align}
    ```

    Example:
    ```pycon
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp
        >>>
        >>> prior = gpx.gps.Prior(
                mean_function = gpx.mean_functions.Zero(),
                kernel = gpx.kernels.RBF()
            )
        >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=100)
        >>>
        >>> posterior = prior * likelihood
    ```
    """

    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: Dataset,
    ) -> GaussianDistribution:
        r"""Query the predictive posterior distribution.

        Conditional on a training data set, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned function
        can be evaluated at a set of test inputs to compute the corresponding
        predictive density.

        The predictive distribution of a conjugate GP is given by
        $$
            p(\mathbf{f}^{\star}\mid \mathbf{y}) & = \int p(\mathbf{f}^{\star} \mathbf{f} \mid \mathbf{y})\\
            & =\mathcal{N}(\mathbf{f}^{\star} \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}
        $$
        where
        $$
            \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y}  \\
            \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
        $$

        The conditioning set is a GPJax `Dataset` object, whilst predictions
        are made on a regular Jax array.

        Example:
        ```pycon
            >>> import gpjax as gpx
            >>> import jax.numpy as jnp
            >>>
            >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
            >>> ytrain = jnp.sin(xtrain)
            >>> D = gpx.Dataset(X=xtrain, y=ytrain)
            >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
            >>>
            >>> prior = gpx.gps.Prior(mean_function = gpx.mean_functions.Zero(), kernel = gpx.kernels.RBF())
            >>> posterior = prior * gpx.likelihoods.Gaussian(num_datapoints = D.n)
            >>> predictive_dist = posterior(xtest, D)
        ```

        Args:
            test_inputs (Num[Array, "N D"]): A Jax array of test inputs at which the
                predictive distribution is evaluated.
            train_data (Dataset): A `gpx.Dataset` object that contains the input and
                output data used for training dataset.

        Returns:
            GaussianDistribution: A function that accepts an input array and
                returns the predictive distribution as a `GaussianDistribution`.
        """
        # Unpack training data
        x, y = train_data.X, train_data.y

        # Unpack test inputs
        t = test_inputs

        # Observation noise o²
        obs_noise = self.likelihood.obs_stddev.value**2
        mx = self.prior.mean_function(x)

        # Precompute Gram matrix, Kxx, at training inputs, x
        Kxx = self.prior.kernel.gram(x)
        Kxx += I_like(Kxx) * self.jitter

        # Σ = Kxx + Io²
        Sigma = Kxx + I_like(Kxx) * obs_noise
        Sigma = PSD(Sigma)

        mean_t = self.prior.mean_function(t)
        Ktt = self.prior.kernel.gram(t)
        Kxt = self.prior.kernel.cross_covariance(x, t)
        Sigma_inv_Kxt = solve(Sigma, Kxt, Cholesky())

        # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
        mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += I_like(covariance) * self.prior.jitter
        covariance = PSD(covariance)

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)

    def sample_approx(
        self,
        num_samples: int,
        train_data: Dataset,
        key: KeyArray,
        num_features: int | None = 100,
        solver_algorithm: tp.Optional[Algorithm] = Cholesky(),
    ) -> FunctionalSample:
        r"""Draw approximate samples from the Gaussian process posterior.

        Build an approximate sample from the Gaussian process posterior. This method
        provides a function that returns the evaluations of a sample across any given
        inputs.

        Unlike when building approximate samples from a Gaussian process prior, decompositions
        based on Fourier features alone rarely give accurate samples. Therefore, we must also
        include an additional set of features (known as canonical features) to better model the
        transition from Gaussian process prior to Gaussian process posterior. For more details
        see [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309).

        In particular, we approximate the Gaussian processes' posterior as the finite
        feature approximation
        $\hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i + \sum{j=1}^N v_jk(.,x_j)$
        where $\phi_i$ are m features sampled from the Fourier feature decomposition of
        the model's kernel and $k(., x_j)$ are N canonical features. The Fourier
        weights $\theta_i$ are samples from a unit Gaussian. See
        [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309) for expressions
        for the canonical weights $v_j$.

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
            solver_algorithm (Optional[Algorithm], optional): The algorithm to use for the solves of
                the inverse of the covariance matrix. See the
                [CoLA documentation](https://cola.readthedocs.io/en/latest/package/cola.linalg.html#algorithms)
                for which solver to pick. For PSD matrices, CoLA currently recommends Cholesky() for small
                matrices and CG() for larger matrices. Select Auto() to let CoLA decide. Defaults to Cholesky().

        Returns:
            FunctionalSample: A function representing an approximate sample from the Gaussian
            process prior.
        """
        if (not isinstance(num_samples, int)) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        # sample fourier features
        fourier_feature_fn = _build_fourier_features_fn(self.prior, num_features, key)

        # sample fourier weights
        fourier_weights = jr.normal(key, [num_samples, 2 * num_features])  # [B, L]

        # sample weights v for canonical features
        # v = Σ⁻¹ (y + ε - ɸ⍵) for  Σ = Kxx + Io² and ε ᯈ N(0, o²)
        obs_var = self.likelihood.obs_stddev.value**2
        Kxx = self.prior.kernel.gram(train_data.X)  #  [N, N]
        Sigma = Kxx + I_like(Kxx) * (obs_var + self.jitter)  #  [N, N]
        eps = jnp.sqrt(obs_var) * jr.normal(key, [train_data.n, num_samples])  #  [N, B]
        y = train_data.y - self.prior.mean_function(train_data.X)  # account for mean
        Phi = fourier_feature_fn(train_data.X)
        canonical_weights = solve(
            Sigma,
            y + eps - jnp.inner(Phi, fourier_weights),
            solver_algorithm,
        )  #  [N, B]

        def sample_fn(test_inputs: Float[Array, "n D"]) -> Float[Array, "n B"]:
            fourier_features = fourier_feature_fn(test_inputs)  # [n, L]
            weight_space_contribution = jnp.inner(
                fourier_features, fourier_weights
            )  # [n, B]
            canonical_features = self.prior.kernel.cross_covariance(
                test_inputs, train_data.X
            )  # [n, N]
            function_space_contribution = jnp.matmul(
                canonical_features, canonical_weights
            )

            return (
                self.prior.mean_function(test_inputs)
                + weight_space_contribution
                + function_space_contribution
            )

        return sample_fn


class NonConjugatePosterior(AbstractPosterior[P, NGL]):
    r"""A non-conjugate Gaussian process posterior object.

    A Gaussian process posterior object for models where the likelihood is
    non-Gaussian. Unlike the `ConjugatePosterior` object, the
    `NonConjugatePosterior` object does not provide an exact marginal
    log-likelihood function. Instead, the `NonConjugatePosterior` object
    represents the posterior distributions as a function of the model's
    hyperparameters and the latent function. Markov chain Monte Carlo,
    variational inference, or Laplace approximations can then be used to sample
    from, or optimise an approximation to, the posterior distribution.
    """

    latent: nnx.Intermediate[Float[Array, "N 1"]]

    def __init__(
        self,
        prior: P,
        likelihood: NGL,
        latent: tp.Union[Float[Array, "N 1"], Parameter, None] = None,
        jitter: float = 1e-6,
        key: KeyArray = jr.PRNGKey(42),
    ):
        r"""Construct a non-conjugate Gaussian process posterior.

        Args:
            prior (AbstractPrior): The prior distribution.
            likelihood (AbstractLikelihood): The likelihood distribution.
            jitter (float): A small constant added to the diagonal of the
                covariance matrix to ensure numerical stability.
        """
        super().__init__(prior=prior, likelihood=likelihood, jitter=jitter)

        if latent is None:
            latent = jr.normal(key, shape=(self.likelihood.num_datapoints, 1))

        # TODO: static or intermediate?
        self.latent = latent if isinstance(latent, Parameter) else Real(latent)
        self.key = Static(key)

    def predict(
        self, test_inputs: Num[Array, "N D"], train_data: Dataset
    ) -> GaussianDistribution:
        r"""Query the predictive posterior distribution.

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
                a `dx.Distribution`.
        """
        # Unpack training data
        x = train_data.X

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # Precompute lower triangular of Gram matrix, Lx, at training inputs, x
        Kxx = kernel.gram(x)
        Kxx += I_like(Kxx) * self.prior.jitter
        Kxx = PSD(Kxx)
        Lx = lower_cholesky(Kxx)

        # Unpack test inputs
        t = test_inputs

        # Compute terms of the posterior predictive distribution
        Ktx = kernel.cross_covariance(t, x)
        Ktt = kernel.gram(t)
        mean_t = mean_function(t)

        # Lx⁻¹ Kxt
        Lx_inv_Kxt = solve(Lx, Ktx.T, Cholesky())

        # Whitened function values, wx, corresponding to the inputs, x
        wx = self.latent.value

        # μt + Ktx Lx⁻¹ wx
        mean = mean_t + jnp.matmul(Lx_inv_Kxt.T, wx)

        # Ktt - Ktx Kxx⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Lx_inv_Kxt.T, Lx_inv_Kxt)
        covariance += I_like(covariance) * self.prior.jitter
        covariance = PSD(covariance)

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)


#######################
# Utils
#######################


@tp.overload
def construct_posterior(prior: P, likelihood: GL) -> ConjugatePosterior[P, GL]: ...


@tp.overload
def construct_posterior(  # noqa: F811
    prior: P, likelihood: NGL
) -> NonConjugatePosterior[P, NGL]: ...


def construct_posterior(prior, likelihood):  # noqa: F811
    r"""Utility function for constructing a posterior object from a prior and
    likelihood. The function will automatically select the correct posterior
    object based on the likelihood.

    Args:
        prior (Prior): The Prior distribution.
        likelihood (AbstractLikelihood): The likelihood that represents our
            beliefs around the distribution of the data.

    Returns
    -------
        AbstractPosterior: A posterior distribution. If the likelihood is
            Gaussian, then a `ConjugatePosterior` will be returned. Otherwise,
            a `NonConjugatePosterior` will be returned.
    """
    if isinstance(likelihood, Gaussian):
        return ConjugatePosterior(prior=prior, likelihood=likelihood)

    return NonConjugatePosterior(prior=prior, likelihood=likelihood)


def _build_fourier_features_fn(
    prior: Prior, num_features: int, key: KeyArray
) -> tp.Callable[[Float[Array, "N D"]], Float[Array, "N L"]]:
    r"""Return a function that evaluates features sampled from the Fourier feature
    decomposition of the prior's kernel.

    Args:
        prior (Prior): The Prior distribution.
        num_features (int): The number of feature functions to be sampled.
        key (KeyArray): The random seed used.

    Returns
    -------
        Callable: A callable function evaluation the sampled feature functions.
    """
    if (not isinstance(num_features, int)) or num_features <= 0:
        raise ValueError("num_features must be a positive integer")

    # Approximate kernel with feature decomposition
    approximate_kernel = RFF(
        base_kernel=prior.kernel, num_basis_fns=num_features, key=key
    )

    def eval_fourier_features(test_inputs: Float[Array, "N D"]) -> Float[Array, "N L"]:
        Phi = approximate_kernel.compute_features(x=test_inputs)
        Phi *= jnp.sqrt(prior.kernel.variance.value / num_features)
        return Phi

    return eval_fourier_features


__all__ = [
    "AbstractPrior",
    "Prior",
    "AbstractPosterior",
    "ConjugatePosterior",
    "NonConjugatePosterior",
    "construct_posterior",
]
