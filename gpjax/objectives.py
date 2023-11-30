from abc import abstractmethod
from dataclasses import dataclass

from jax import vmap
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jaxtyping import Float
import tensorflow_probability.substrates.jax as tfp

from gpjax.base import (
    Module,
    static_field,
)
from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.lower_cholesky import lower_cholesky
from gpjax.typing import (
    Array,
    ScalarFloat,
)

tfd = tfp.distributions

from typing import TypeVar

import cola

ConjugatePosterior = TypeVar(
    "ConjugatePosterior", bound="gpjax.gps.ConjugatePosterior"  # noqa: F821
)
NonConjugatePosterior = TypeVar(
    "NonConjugatePosterior", bound="gpjax.gps.NonConjugatePosterior"  # noqa: F821
)
VariationalFamily = TypeVar(
    "VariationalFamily",
    bound="gpjax.variational_families.AbstractVariationalFamily",  # noqa: F821
)

from cola.linalg.decompositions.decompositions import Cholesky


@dataclass
class AbstractObjective(Module):
    r"""Abstract base class for objectives."""

    negative: bool = static_field(False)
    constant: ScalarFloat = static_field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.constant = jnp.array(-1.0) if self.negative else jnp.array(1.0)

    def __hash__(self):
        return hash(tuple(jtu.tree_leaves(self)))  # Probably put this on the Module!

    def __call__(self, *args, **kwargs) -> ScalarFloat:
        return self.step(*args, **kwargs)

    @abstractmethod
    def step(self, *args, **kwargs) -> ScalarFloat:
        raise NotImplementedError


class ConjugateMLL(AbstractObjective):
    def step(
        self,
        posterior: ConjugatePosterior,
        train_data: Dataset,
    ) -> ScalarFloat:
        r"""Evaluate the marginal log-likelihood of the Gaussian process.

        Compute the marginal log-likelihood function of the Gaussian process.
        The returned function can then be used for gradient based optimisation
        of the model's parameters or for model comparison. The implementation
        given here enables exact estimation of the Gaussian process' latent
        function values.

        For a training dataset $`\{x_n, y_n\}_{n=1}^N`$, set of test inputs
        $`\mathbf{x}^{\star}`$ the corresponding latent function evaluations are given
        by $`\mathbf{f}=f(\mathbf{x})`$ and $`\mathbf{f}^{\star}f(\mathbf{x}^{\star})`$,
        the marginal log-likelihood is given by:
        ```math
        \begin{align}
            \log p(\mathbf{y}) & = \int p(\mathbf{y}\mid\mathbf{f})p(\mathbf{f}, \mathbf{f}^{\star}\mathrm{d}\mathbf{f}^{\star}\\
            &=0.5\left(-\mathbf{y}^{\top}\left(k(\mathbf{x}, \mathbf{x}') +\sigma^2\mathbf{I}_N  \right)^{-1}\mathbf{y}-\log\lvert k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_N\rvert - n\log 2\pi \right).
        \end{align}
        ```

        For a given ``ConjugatePosterior`` object, the following code snippet shows
        how the marginal log-likelihood can be evaluated.

        Example:
        ```python
            >>> import gpjax as gpx
            >>>
            >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
            >>> ytrain = jnp.sin(xtrain)
            >>> D = gpx.Dataset(X=xtrain, y=ytrain)
            >>>
            >>> meanf = gpx.mean_functions.Constant()
            >>> kernel = gpx.kernels.RBF()
            >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
            >>> prior = gpx.gps.Prior(mean_function = meanf, kernel=kernel)
            >>> posterior = prior * likelihood
            >>>
            >>> mll = gpx.objectives.ConjugateMLL(negative=True)
            >>> mll(posterior, train_data = D)
        ```

        Our goal is to maximise the marginal log-likelihood. Therefore, when optimising
        the model's parameters with respect to the parameters, we use the negative
        marginal log-likelihood. This can be realised through

        ```python
            mll = gpx.objectives.ConjugateMLL(negative=True)
        ```

        For optimal performance, the marginal log-likelihood should be ``jax.jit``
        compiled.
        ```python
            mll = jit(gpx.objectives.ConjugateMLL(negative=True))
        ```

        Args:
            posterior (ConjugatePosterior): The posterior distribution for which
                we want to compute the marginal log-likelihood.
            train_data (Dataset): The training dataset used to compute the
                marginal log-likelihood.

        Returns
        -------
            ScalarFloat: The marginal log-likelihood of the Gaussian process for the
                current parameter set.
        """
        x, y = train_data.X, train_data.y

        # Observation noise o²
        obs_noise = posterior.likelihood.obs_stddev**2
        mx = posterior.prior.mean_function(x)

        # Σ = (Kxx + Io²) = LLᵀ
        Kxx = posterior.prior.kernel.gram(x)
        Kxx += cola.ops.I_like(Kxx) * posterior.prior.jitter
        Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
        Sigma = cola.PSD(Sigma)

        # p(y | x, θ), where θ are the model hyperparameters:
        mll = GaussianDistribution(jnp.atleast_1d(mx.squeeze()), Sigma)

        return self.constant * (mll.log_prob(jnp.atleast_1d(y.squeeze())).squeeze())


class ConjugateLOOCV(AbstractObjective):
    def step(
        self,
        posterior: ConjugatePosterior,
        train_data: Dataset,
    ) -> ScalarFloat:
        r"""Evaluate the leave-one-out log predictive probability of the Gaussian process following
        section 5.4.2 of Rasmussen et al. 2006 - Gaussian Processes for Machine Learning. This metric
        calculates the average performance of all models that can be obtained by training on all but one
        data point, and then predicting the left out data point.

        The returned metric can then be used for gradient based optimisation
        of the model's parameters or for model comparison. The implementation
        given here enables exact estimation of the Gaussian process' latent
        function values.

        For a given ``ConjugatePosterior`` object, the following code snippet shows
        how the leave-one-out log predicitive probability can be evaluated.

        Example:
        ```python
            >>> import gpjax as gpx
            >>>
            >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
            >>> ytrain = jnp.sin(xtrain)
            >>> D = gpx.Dataset(X=xtrain, y=ytrain)
            >>>
            >>> meanf = gpx.mean_functions.Constant()
            >>> kernel = gpx.kernels.RBF()
            >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
            >>> prior = gpx.gps.Prior(mean_function = meanf, kernel=kernel)
            >>> posterior = prior * likelihood
            >>>
            >>> loocv = gpx.objectives.ConjugateLOOCV(negative=True)
            >>> loocv(posterior, train_data = D)
        ```

        Our goal is to maximise the leave-one-out log predictive probability. Therefore, when
        optimising the model's parameters with respect to the parameters, we use the negative
        leave-one-out log predictive probability. This can be realised through

        ```python
            mll = gpx.objectives.ConjugateLOOCV(negative=True)
        ```

        For optimal performance, the objective should be ``jax.jit``
        compiled.
        ```python
            mll = jit(gpx.objectives.ConjugateLOOCV(negative=True))
        ```

        Args:
            posterior (ConjugatePosterior): The posterior distribution for which
                we want to compute the leave-one-out log predictive probability.
            train_data (Dataset): The training dataset used to compute the
                leave-one-out log predictive probability..

        Returns
        -------
            ScalarFloat: The leave-one-out log predictive probability of the Gaussian
                process for the current parameter set.
        """
        x, y = train_data.X, train_data.y
        y.shape[1]

        # Observation noise o²
        obs_var = posterior.likelihood.obs_stddev**2

        mx = posterior.prior.mean_function(x)  # [N, M]

        # Σ = (Kxx + Io²)
        Kxx = posterior.prior.kernel.gram(x)
        Sigma = Kxx + cola.ops.I_like(Kxx) * (obs_var + posterior.prior.jitter)
        Sigma = cola.PSD(Sigma)  # [N, N]

        Sigma_inv_y = cola.solve(Sigma, y - mx, Cholesky())  # [N, 1]
        Sigma_inv_diag = cola.linalg.diag(cola.inv(Sigma, Cholesky()))[
            :, None
        ]  # [N, 1]

        loocv_means = mx + (y - mx) - Sigma_inv_y / Sigma_inv_diag
        loocv_stds = jnp.sqrt(1.0 / Sigma_inv_diag)

        loocv_posterior = tfd.Normal(loc=loocv_means, scale=loocv_stds)
        loocv = jnp.sum(loocv_posterior.log_prob(y))
        return self.constant * loocv


class LogPosteriorDensity(AbstractObjective):
    r"""The log-posterior density of a non-conjugate Gaussian process. This is
    sometimes referred to as the marginal log-likelihood.
    """

    def step(self, posterior: NonConjugatePosterior, data: Dataset) -> ScalarFloat:
        r"""Evaluate the log-posterior density of a Gaussian process.

        Compute the marginal log-likelihood, or log-posterior density of the Gaussian
        process. The returned function can then be used for gradient based optimisation
        of the model's parameters or for model comparison. The implementation given
        here is general and will work for any likelihood support by GPJax.

        Unlike the marginal_log_likelihood function of the `ConjugatePosterior` object,
        the marginal_log_likelihood function of the `NonConjugatePosterior` object does
        not provide an exact marginal log-likelihood function. Instead, the
        `NonConjugatePosterior` object represents the posterior distributions as a
        function of the model's hyperparameters and the latent function. Markov chain
        Monte Carlo, variational inference, or Laplace approximations can then be used
        to sample from, or optimise an approximation to, the posterior distribution.

        Args:
            posterior (NonConjugatePosterior): The posterior distribution for which
                we want to compute the marginal log-likelihood.
            data (Dataset): The training dataset used to compute the
                marginal log-likelihood.

        Returns
        -------
            ScalarFloat: The log-posterior density of the Gaussian process for the
                current parameter set.
        """
        # Unpack the training data
        x, y = data.X, data.y
        Kxx = posterior.prior.kernel.gram(x)
        Kxx += cola.ops.I_like(Kxx) * posterior.prior.jitter
        Kxx = cola.PSD(Kxx)
        Lx = lower_cholesky(Kxx)

        # Compute the prior mean function
        mx = posterior.prior.mean_function(x)

        # Whitened function values, wx, corresponding to the inputs, x
        wx = posterior.latent

        # f(x) = mx  +  Lx wx
        fx = mx + Lx @ wx

        # p(y | f(x), θ), where θ are the model hyperparameters
        likelihood = posterior.likelihood.link_function(fx)

        # Whitened latent function values prior, p(wx | θ) = N(0, I)
        latent_prior = tfd.Normal(loc=0.0, scale=1.0)

        return self.constant * (
            likelihood.log_prob(y).sum() + latent_prior.log_prob(wx).sum()
        )


NonConjugateMLL = LogPosteriorDensity


class ELBO(AbstractObjective):
    def step(
        self,
        variational_family: VariationalFamily,
        train_data: Dataset,
    ) -> ScalarFloat:
        r"""Compute the evidence lower bound of a variational approximation.

        Compute the evidence lower bound under this model. In short, this requires
        evaluating the expectation of the model's log-likelihood under the variational
        approximation. To this, we sum the KL divergence from the variational posterior
        to the prior. When batching occurs, the result is scaled by the batch size
        relative to the full dataset size.

        Args:
            variational_family (AbstractVariationalFamily): The variational
                approximation for whose parameters we should maximise the ELBO with
                respect to.
            train_data (Dataset): The training data for which we should maximise the
                ELBO with respect to.

        Returns
        -------
            ScalarFloat: The evidence lower bound of the variational approximation for
                the current model parameter set.
        """
        # KL[q(f(·)) || p(f(·))]
        kl = variational_family.prior_kl()

        # ∫[log(p(y|f(·))) q(f(·))] df(·)
        var_exp = variational_expectation(variational_family, train_data)

        # For batch size b, we compute  n/b * Σᵢ[ ∫log(p(y|f(xᵢ))) q(f(xᵢ)) df(xᵢ)] - KL[q(f(·)) || p(f(·))]
        return self.constant * (
            jnp.sum(var_exp)
            * variational_family.posterior.likelihood.num_datapoints
            / train_data.n
            - kl
        )


def variational_expectation(
    variational_family: VariationalFamily,
    train_data: Dataset,
) -> Float[Array, " N"]:
    r"""Compute the variational expectation.

    Compute the expectation of our model's log-likelihood under our variational
    distribution. Batching can be done here to speed up computation.

    Args:
        variational_family (AbstractVariationalFamily): The variational family that we
            are using to approximate the posterior.
        train_data (Dataset): The batch for which the expectation should be computed
            for.

    Returns
    -------
        Array: The expectation of the model's log-likelihood under our variational
            distribution.
    """
    # Unpack training batch
    x, y = train_data.X, train_data.y

    # Variational distribution q(f(·)) = N(f(·); μ(·), Σ(·, ·))
    q = variational_family

    # TODO: This needs cleaning up! We are squeezing then broadcasting `mean` and `variance`, which is not ideal.

    # Compute variational mean, μ(x), and variance, diag(Σ(x, x)), at the training
    # inputs, x
    def q_moments(x):
        qx = q(x)
        return qx.mean().squeeze(), qx.covariance().squeeze()

    mean, variance = vmap(q_moments)(x[:, None])

    # ≈ ∫[log(p(y|f(x))) q(f(x))] df(x)
    expectation = q.posterior.likelihood.expected_log_likelihood(
        y, mean[:, None], variance[:, None]
    )
    return expectation


# TODO: Replace code within CollapsedELBO to using (low rank structure of) LinOps and the GaussianDistribution object to be as succinct as e.g., the `ConjugateMLL`.


class CollapsedELBO(AbstractObjective):
    r"""The collapsed evidence lower bound.

    Collapsed variational inference for a sparse Gaussian process regression model.
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables
    in Sparse Gaussian Processes.
    """

    def step(
        self,
        variational_family: VariationalFamily,
        train_data: Dataset,
    ) -> ScalarFloat:
        r"""Compute a single step of the collapsed evidence lower bound.

        Compute the evidence lower bound under this model. In short, this requires
        evaluating the expectation of the model's log-likelihood under the variational
        approximation. To this, we sum the KL divergence from the variational posterior
        to the prior. When batching occurs, the result is scaled by the batch size
        relative to the full dataset size.

        Args:
            variational_family (AbstractVariationalFamily): The variational
                approximation for whose parameters we should maximise the ELBO with
                respect to.
            train_data (Dataset): The training data for which we should maximise the
                ELBO with respect to.

        Returns
        -------
            ScalarFloat: The evidence lower bound of the variational approximation for
                the current model parameter set.
        """
        # Unpack training data
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack mean function and kernel
        mean_function = variational_family.posterior.prior.mean_function
        kernel = variational_family.posterior.prior.kernel

        m = variational_family.num_inducing

        noise = variational_family.posterior.likelihood.obs_stddev**2
        z = variational_family.inducing_inputs
        Kzz = kernel.gram(z)
        Kzz += cola.ops.I_like(Kzz) * variational_family.jitter
        Kzz = cola.PSD(Kzz)
        Kzx = kernel.cross_covariance(z, x)
        Kxx_diag = vmap(kernel, in_axes=(0, 0))(x, x)
        μx = mean_function(x)

        Lz = lower_cholesky(Kzz)

        # Notation and derivation:
        #
        # Let Q = KxzKzz⁻¹Kzx, we must compute the log normal pdf:
        #
        #   log N(y; μx, o²I + Q) = -nπ - n/2 log|o²I + Q|
        #   - 1/2 (y - μx)ᵀ (o²I + Q)⁻¹ (y - μx).
        #
        # The log determinant |o²I + Q| is computed via applying the matrix determinant
        #   lemma
        #
        #   |o²I + Q| = log|o²I| + log|I + Lz⁻¹ Kzx (o²I)⁻¹ Kxz Lz⁻¹| = log(o²) +  log|B|,
        #
        #   with B = I + AAᵀ and A = Lz⁻¹ Kzx / o.
        #
        # Similarly we apply matrix inversion lemma to invert o²I + Q
        #
        #   (o²I + Q)⁻¹ = (Io²)⁻¹ - (Io²)⁻¹ Kxz Lz⁻ᵀ (I + Lz⁻¹ Kzx (Io²)⁻¹ Kxz Lz⁻ᵀ )⁻¹ Lz⁻¹ Kzx (Io²)⁻¹
        #               = (Io²)⁻¹ - (Io²)⁻¹ oAᵀ (I + oA (Io²)⁻¹ oAᵀ)⁻¹ oA (Io²)⁻¹
        #               = I/o² - Aᵀ B⁻¹ A/o²,
        #
        # giving the quadratic term as
        #
        #   (y - μx)ᵀ (o²I + Q)⁻¹ (y - μx) = [(y - μx)ᵀ(y - µx)  - (y - μx)ᵀ Aᵀ B⁻¹ A (y - μx)]/o²,
        #
        #   with A and B defined as above.

        A = cola.solve(Lz, Kzx, Cholesky()) / jnp.sqrt(noise)

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

        # (y - μx)ᵀ (Io² + Q)⁻¹ (y - μx)
        quad = (jnp.sum(diff**2) - jnp.sum(L_inv_A_diff**2)) / noise

        # 2 * log N(y; μx, Io² + Q)
        two_log_prob = -n * jnp.log(2.0 * jnp.pi * noise) - log_det_B - quad

        # 1/o² tr(Kxx - Q) [Trace law tr(AB) = tr(BA) => tr(KxzKzz⁻¹Kzx) = tr(KxzLz⁻ᵀLz⁻¹Kzx) = tr(Lz⁻¹Kzx KxzLz⁻ᵀ) = trace(o²AAᵀ)]
        two_trace = jnp.sum(Kxx_diag) / noise - jnp.trace(AAT)

        # log N(y; μx, Io² + KxzKzz⁻¹Kzx) - 1/2o² tr(Kxx - KxzKzz⁻¹Kzx)
        return self.constant * (two_log_prob - two_trace).squeeze() / 2.0
