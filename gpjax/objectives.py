from .gps import AbstractPosterior, ConjugatePosterior, NonConjugatePosterior
from jaxutils import Dataset, Parameters
from jaxtyping import Float, Array
from abc import ABCMeta, abstractmethod
from jaxlinop import identity
from gpjax.gaussian_distribution import GaussianDistribution
import jax.numpy as jnp
import distrax as dx
from .likelihoods import NonConjugate


class AbstractObjective(metaclass=ABCMeta):
    def __init__(
        self, model: AbstractPosterior, negative: bool, name: str = "Abstract Objective"
    ) -> None:
        self.model = model
        self.constant = jnp.array(-1.0) if negative else jnp.array(1.0)
        self.name = name
        self.jitter = self.model.jitter

    @abstractmethod
    def __call__(
        self, params: Parameters, data: Dataset, **kwargs
    ) -> Float[Array, "1"]:
        raise NotImplementedError(f"__call__ method not implemented for {self.name}.")


class ConjugateMarginalLogLikelihood(AbstractObjective):
    def __init__(
        self,
        model: ConjugatePosterior,
        negative: bool,
        name: str = "Conjugate Marginal Log Likelihood",
    ) -> None:
        if isinstance(model.likelihood, NonConjugate):
            raise ValueError(
                f"ConjugateMarginalLogLikelihood objective can only be used with conjugate likelihoods. {model.likelihood} is not conjugate to Gaussian distribution."
            )
        super().__init__(model, negative, name)

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
            Callable[[Dict], Float[Array, "1"]]: A functional representation
                of the marginal log-likelihood that can be evaluated at a
                given parameter set.
        """

        x, y, n = data.X, data.y, data.n

        # Observation noise σ²
        obs_noise = params["likelihood"]["obs_noise"]
        μx = self.model.prior.mean_function(params["mean_function"], x)

        # Σ = (Kxx + Iσ²) = LLᵀ
        Kxx = self.model.prior.kernel.gram(params["kernel"], x)
        Kxx += identity(n) * self.jitter
        Sigma = Kxx + identity(n) * obs_noise

        # p(y | x, θ), where θ are the model hyperparameters:
        mll = GaussianDistribution(jnp.atleast_1d(μx.squeeze()), Sigma)

        return self.constant * (mll.log_prob(jnp.atleast_1d(y.squeeze())).squeeze())


class NonConjugateMarginalLogLikelihood(AbstractObjective):
    def __init__(
        self,
        model: NonConjugatePosterior,
        negative: bool,
        name: str = "Non-conjugate Marginal Log Likelihood",
    ) -> None:
        super().__init__(model, negative, name)

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
            Callable[[Dict], Float[Array, "1"]]: A functional representation
                of the marginal log-likelihood that can be evaluated at a given
                parameter set.
        """
        # Unpack the training data
        x, y, n = data.X, data.y, data.n
        Kxx = self.model.prior.kernel.gram(params["kernel"], x)
        Kxx += identity(n) * self.jitter
        Lx = Kxx.to_root()

        # Compute the prior mean function
        μx = self.model.prior.mean_function(params["mean_function"], x)

        # Whitened function values, wx, corresponding to the inputs, x
        wx = params["latent"]

        # f(x) = μx  +  Lx wx
        fx = μx + Lx @ wx

        # p(y | f(x), θ), where θ are the model hyperparameters
        likelihood = self.model.likelihood.link_function(params, fx)

        # Whitened latent function values prior, p(wx | θ) = N(0, I)
        latent_prior = dx.Normal(loc=0.0, scale=1.0)

        return self.constant * (
            likelihood.log_prob(y).sum() + latent_prior.log_prob(wx).sum()
        )
