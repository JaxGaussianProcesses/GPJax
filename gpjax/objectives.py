from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gps import AbstractPosterior, ConjugatePosterior, NonConjugatePosterior
    from .variational_families import AbstractVariationalFamily

from abc import ABCMeta, abstractmethod
from typing import Dict

import distrax as dx
from jax import vmap
import jax.numpy as jnp
from jax.random import KeyArray
import jax.scipy as jsp
from jaxlinop import identity
from jaxtyping import Array, Float
from jaxutils import Dataset, Parameters

from .gaussian_distribution import GaussianDistribution
from .likelihoods import AbstractLikelihood, Gaussian, NonConjugate
from .quadrature import gauss_hermite_quadrature


class AbstractObjective(metaclass=ABCMeta):
    def __init__(
        self,
        posterior:"AbstractPosterior",
        negative: bool,
        name: str = "Abstract Objective",
    ) -> None:
        self.posterior = posterior
        self.constant = jnp.array(-1.0) if negative else jnp.array(1.0)
        self.name = name
        self.jitter = self.posterior.jitter

    @abstractmethod
    def __call__(
        self, params: Parameters, data: Dataset, **kwargs
    ) -> Float[Array, "1"]:
        raise NotImplementedError(f"__call__ method not implemented for {self.name}.")

    @abstractmethod
    def init_params(self, key: KeyArray) -> Dict:
        raise NotImplementedError(
            f"init_params method not implemented for {self.name}."
        )


class ConjugateMLL(AbstractObjective):
    def __init__(
        self,
        posterior:"ConjugatePosterior",
        negative: bool,
        name: str = "Conjugate Marginal Log Likelihood",
    ) -> None:
        if isinstance(posterior.likelihood, NonConjugate):
            raise ValueError(
                f"""ConjugateMLL objective can only be used with
                conjugate likelihoods. {posterior.likelihood} is not conjugate to Gaussian
                distribution."""
            )
        super().__init__(posterior, negative, name)

    def __call__(
        self, params: Parameters, data: Dataset, **kwargs
    ) -> Float[Array, "1"]:
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
            negative (Optional[bool]): Whether or not the returned function
                should be negative. For optimisation, the negative is useful
                as minimisation of the negative marginal log-likelihood is
                equivalent to maximisation of the marginal log-likelihood.
                Defaults to False.

        Returns:
            Callable[[Parameters], Float[Array, "1"]]: A functional representation
                of the marginal log-likelihood that can be evaluated at a
                given parameter set.
        """

        x, y, n = data.X, data.y, data.n

        # Observation noise σ²
        obs_noise = params["likelihood"]["obs_noise"]
        μx = self.posterior.prior.mean_function(params["mean_function"], x)

        # Σ = (Kxx + Iσ²) = LLᵀ
        Kxx = self.posterior.prior.kernel.gram(params["kernel"], x)
        Kxx += identity(n) * self.jitter
        Sigma = Kxx + identity(n) * obs_noise

        # p(y | x, θ), where θ are the model hyperparameters:
        mll = GaussianDistribution(jnp.atleast_1d(μx.squeeze()), Sigma)

        return self.constant * (mll.log_prob(jnp.atleast_1d(y.squeeze())).squeeze())

    def init_params(self, key: KeyArray) -> Dict:
        return self.posterior.init_params(key)


class NonConjugateMLL(AbstractObjective):
    def __init__(
        self,
        posterior:"NonConjugatePosterior",
        negative: bool,
        name: str = "Non-conjugate Marginal Log Likelihood",
    ) -> None:
        super().__init__(posterior, negative, name)

    def __call__(
        self, params: Parameters, data: Dataset, **kwargs
    ) -> Float[Array, "1"]:
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
            Callable[[Parameters], Float[Array, "1"]]: A functional representation
                of the marginal log-likelihood that can be evaluated at a given
                parameter set.
        """
        # Unpack the training data
        x, y, n = data.X, data.y, data.n
        Kxx = self.posterior.prior.kernel.gram(params["kernel"], x)
        Kxx += identity(n) * self.jitter
        Lx = Kxx.to_root()

        # Compute the prior mean function
        μx = self.posterior.prior.mean_function(params["mean_function"], x)

        # Whitened function values, wx, corresponding to the inputs, x
        wx = params["latent"]

        # f(x) = μx  +  Lx wx
        fx = μx + Lx @ wx

        # p(y | f(x), θ), where θ are the model hyperparameters
        likelihood = self.posterior.likelihood.link_function(params, fx)

        # Whitened latent function values prior, p(wx | θ) = N(0, I)
        latent_prior = dx.Normal(loc=0.0, scale=1.0)

        return self.constant * (
            likelihood.log_prob(y).sum() + latent_prior.log_prob(wx).sum()
        )

    def init_params(self, key: KeyArray) -> Dict:
        return self.posterior.init_params(key)


class ELBO(AbstractObjective):
    def __init__(
        self,
        num_datapoints: int,
        posterior: "AbstractPosterior",
        variational_family: "AbstractVariationalFamily",
        negative: bool,
        name: str = "Evidence lower bound",
    ) -> None:

        super().__init__(variational_family, negative, name)
        self.num_datapoints = num_datapoints
        self.posterior = posterior
        self.variational_family = variational_family
        self.negative = negative
        self.name = name
        self.constant = jnp.array(-1.0) if self.negative else jnp.array(1.0)

    def __call__(self, params: Parameters, train_data: Dataset) -> Float[Array, "1"]:
        """Compute the evidence lower bound under this model. In short, this requires
        evaluating the expectation of the model's log-likelihood under the variational
        approximation. To this, we sum the KL divergence from the variational posterior
        to the prior. When batching occurs, the result is scaled by the batch size
        relative to the full dataset size.

        Args:
            params (Parameters): The set of parameters that induce our variational
                approximation.
            train_data (Dataset): The training data for which we should maximise the
                ELBO with respect to.
            negative (bool, optional): Whether or not the resultant elbo function should
                be negative. For gradient descent where we minimise our objective
                function this argument should be true as minimisation of the negative
                corresponds to maximisation of the ELBO. Defaults to False.

        Returns:
            Callable[[Parameters, Dataset], Array]: A callable function that accepts a
                current parameter estimate and batch of data for which gradients should
                be computed.
        """

        # KL[q(f(·)) || p(f(·))]
        kl = self.variational_family.prior_kl(params)

        # ∫[log(p(y|f(·))) q(f(·))] df(·)
        var_exp = variational_expectation(
            params, self.variational_family, self.posterior.likelihood, train_data
        )

        # For batch size b, we compute  n/b * Σᵢ[ ∫log(p(y|f(xᵢ))) q(f(xᵢ)) df(xᵢ)] - KL[q(f(·)) || p(f(·))]
        return self.constant * (
            jnp.sum(var_exp) * self.num_datapoints / train_data.n - kl
        )

    def init_params(self, key: KeyArray) -> Dict:
        """Construct the parameter set used within the variational scheme adopted."""
        variational_params = self.variational_family.init_params(key)
        likelihood_params = self.posterior.likelihood.init_params(key)
        variational_params.add_parameter(
            key="likelihood",
            parameter=likelihood_params,
        )
        return variational_params


LogPosteriorDensity = NonConjugateMLL


def variational_expectation(
    params: Parameters,
    variational_family: "AbstractVariationalFamily",
    likelihood: AbstractLikelihood,
    train_data: Dataset,
) -> Float[Array, "N 1"]:
    """Compute the expectation of our model's log-likelihood under our variational
    distribution. Batching can be done here to speed up computation.

    Args:
        params (Parameters): The set of parameters that induce our variational
            approximation.
        likelihood (AbstractLikelihood)
        batch (Dataset): The batch for which the expectation should be computed for.

    Returns:
        Array: The expectation of the model's log-likelihood under our variational
            distribution.
    """

    # Unpack training batch
    x, y = train_data.X, train_data.y

    # Variational distribution q(f(·)) = N(f(·); μ(·), Σ(·, ·))
    q = variational_family(params)

    # Compute variational mean, μ(x), and variance, √diag(Σ(x, x)), at the training
    # inputs, x
    def q_moments(x):
        qx = q(x)
        return qx.mean(), qx.variance()

    mean, variance = vmap(q_moments)(x[:, None])

    # log(p(y|f(x)))
    link_function = likelihood.link_function
    log_prob = vmap(lambda f, y: link_function(params["likelihood"], f).log_prob(y))

    # ≈ ∫[log(p(y|f(x))) q(f(x))] df(x)
    expectation = gauss_hermite_quadrature(log_prob, mean, jnp.sqrt(variance), y=y)

    return expectation


class CollapsedELBO(AbstractObjective):
    """Collapsed variational inference for a sparse Gaussian process regression model.
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables
    in Sparse Gaussian Processes.
    """

    def __init__(
        self,
        posterior: "AbstractPosterior",
        variational_family: "AbstractVariationalFamily",
        negative: bool,
        name: str = "Conjugate Marginal Log Likelihood",
    ) -> None:
        """Initialise the variational inference module.

        Args:
            posterior (AbstractPosterior): The exact posterior distribution.
            variational_family (AbstractVariationalFamily): The variational family to
                be trained.
        """

        if not isinstance(posterior.likelihood, Gaussian):
            raise TypeError("Likelihood must be Gaussian.")

        super().__init__(variational_family, negative, name)
        self.posterior = posterior
        self.variational_family = variational_family

    def __call__(self, params: Parameters, train_data: Dataset) -> Float[Array, "1"]:
        """Compute the evidence lower bound under this model. In short, this requires
        evaluating the expectation of the model's log-likelihood under the variational
        approximation. To this, we sum the KL divergence from the variational posterior
        to the prior. When batching occurs, the result is scaled by the batch size
        relative to the full dataset size.

        Args:
            train_data (Dataset): The training data for which we should maximise the
                ELBO with respect to.
            negative (bool, optional): Whether or not the resultant elbo function should
                be negative. For gradient descent where we minimise our objective
                function this argument should be true as minimisation of the negative
                corresponds to maximisation of the ELBO. Defaults to False.

        Returns:
            Callable[[Parameters, Dataset], Array]: A callable function that accepts a
                current parameter estimate for which gradients should be computed.
        """

        # Unpack training data
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        m = self.variational_family.num_inducing

        noise = params["likelihood"]["obs_noise"]
        z = params["variational_family"]["inducing_inputs"]
        Kzz = kernel.gram(params["kernel"], z)
        Kzz += identity(m) * self.jitter
        Kzx = kernel.cross_covariance(params["kernel"], z, x)
        Kxx_diag = vmap(kernel, in_axes=(None, 0, 0))(params["kernel"], x, x)
        μx = mean_function(params["mean_function"], x)

        Lz = Kzz.to_root()

        # Notation and derivation:
        #
        # Let Q = KxzKzz⁻¹Kzx, we must compute the log normal pdf:
        #
        #   log N(y; μx, σ²I + Q) = -nπ - n/2 log|σ²I + Q|
        #   - 1/2 (y - μx)ᵀ (σ²I + Q)⁻¹ (y - μx).
        #
        # The log determinant |σ²I + Q| is computed via applying the matrix determinant
        #   lemma
        #
        #   |σ²I + Q| = log|σ²I| + log|I + Lz⁻¹ Kzx (σ²I)⁻¹ Kxz Lz⁻¹| = log(σ²) +  log|B|,
        #
        #   with B = I + AAᵀ and A = Lz⁻¹ Kzx / σ.
        #
        # Similarly we apply matrix inversion lemma to invert σ²I + Q
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
        L_inv_A_diff = jsp.linalg.solve_triangular(L, jnp.matmul(A, diff), lower=True)

        # (y - μx)ᵀ (Iσ² + Q)⁻¹ (y - μx)
        quad = (jnp.sum(diff**2) - jnp.sum(L_inv_A_diff**2)) / noise

        # 2 * log N(y; μx, Iσ² + Q)
        two_log_prob = -n * jnp.log(2.0 * jnp.pi * noise) - log_det_B - quad

        # 1/σ² tr(Kxx - Q) [Trace law tr(AB) = tr(BA) => tr(KxzKzz⁻¹Kzx) = tr(KxzLz⁻ᵀLz⁻¹Kzx) = tr(Lz⁻¹Kzx KxzLz⁻ᵀ) = trace(σ²AAᵀ)]
        two_trace = jnp.sum(Kxx_diag) / noise - jnp.trace(AAT)

        # log N(y; μx, Iσ² + KxzKzz⁻¹Kzx) - 1/2σ² tr(Kxx - KxzKzz⁻¹Kzx)
        return self.constant * (two_log_prob - two_trace).squeeze() / 2.0

    def init_params(self, key: KeyArray) -> Dict:
        """Construct the parameter set used within the variational scheme adopted."""
        return self.variational_family.init_params(key)
