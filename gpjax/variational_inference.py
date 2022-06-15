import abc
from typing import Callable, Dict

import jax.numpy as jnp
import jax.scipy as jsp
from chex import dataclass
from jax import vmap

from .gps import AbstractPosterior
from .parameters import transform
from .quadrature import gauss_hermite_quadrature
from .types import Array, Dataset
from .utils import concat_dictionaries, I
from .variational_families import AbstractVariationalFamily
from .kernels import cross_covariance, diagonal, gram
from .likelihoods import Gaussian
from .variational_families import CollapsedVariationalGaussian

@dataclass
class AbstractVariationalInference:
    """A base class for inference and training of variational families against an extact posterior"""

    posterior: AbstractPosterior
    variational_family: AbstractVariationalFamily

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood

    @property
    def params(self) -> Dict:
        """Construct the parameter set used within the variational scheme adopted."""
        hyperparams = concat_dictionaries(
            {"likelihood": self.posterior.likelihood.params}, 
            self.variational_family.params,
        )
        return hyperparams

    @abc.abstractmethod
    def elbo(
        self, train_data: Dataset, transformations: Dict
    ) -> Callable[[Dict], Array]:
        """Placeholder method for computing the evidence lower bound function (ELBO), given a training dataset and a set of transformations that map each parameter onto the entire real line.

        Args:
            train_data (Dataset): The training dataset for which the ELBO is to be computed.
            transformations (Dict): A set of functions that unconstrain each parameter.

        Returns:
            Callable[[Array], Array]: A function that computes the ELBO given a set of parameters.
        """
        raise NotImplementedError


@dataclass
class StochasticVI(AbstractVariationalInference):
    """Stochastic Variational inference training module. The key reference is Hensman et. al., (2013) - Gaussian processes for big data."""

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood
        self.num_inducing = self.variational_family.num_inducing

    def elbo(
        self, train_data: Dataset, transformations: Dict, negative: bool = False
    ) -> Callable[[Array], Array]:
        """Compute the evidence lower bound under this model. In short, this requires evaluating the expectation of the model's log-likelihood under the variational approximation. To this, we sum the KL divergence from the variational posterior to the prior. When batching occurs, the result is scaled by the batch size relative to the full dataset size.

        Args:
            train_data (Dataset): The training data for which we should maximise the ELBO with respect to.
            transformations (Dict): The transformation set that unconstrains each parameter.
            negative (bool, optional): Whether or not the resultant elbo function should be negative. For gradient descent where we minimise our objective function this argument should be true as minimisation of the negative corresponds to maximisation of the ELBO. Defaults to False.

        Returns:
            Callable[[Dict, Dataset], Array]: A callable function that accepts a current parameter estimate and batch of data for which gradients should be computed.
        """
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def elbo_fn(params: Dict, batch: Dataset) -> Array:
            params = transform(params, transformations)

            # KL[q(f(·)) || p(f(·))]
            kl = self.variational_family.prior_kl(params)

            # ∫[log(p(y|f(x))) q(f(x))] df(x)
            var_exp = self.variational_expectation(params, batch)

            # For batch size b, we compute  n/b * Σᵢ[ ∫log(p(y|f(xᵢ))) q(f(xᵢ)) df(xᵢ)] - KL[q(f(·)) || p(f(·))]
            return constant * (jnp.sum(var_exp) * train_data.n / batch.n - kl)

        return elbo_fn

    def variational_expectation(self, params: Dict, batch: Dataset) -> Array:
        """Compute the expectation of our model's log-likelihood under our variational distribution. Batching can be done here to speed up computation.

        Args:
            params (Dict): The set of parameters that induce our variational approximation.
            batch (Dataset): The data batch for which the expectation should be computed for.

        Returns:
            Array: The expectation of the model's log-likelihood under our variational distribution.
        """
        x, y = batch.X, batch.y

        # q(f(x))
        predictive_dist = vmap(self.variational_family.predict(params))(x[:, None])
        mean = predictive_dist.mean().val.reshape(-1,1)
        variance = predictive_dist.variance().val.reshape(-1,1)

        # log(p(y|f(x)))
        log_prob = vmap(lambda f, y: self.likelihood.link_function(f, params["likelihood"]).log_prob(y))

        # ≈ ∫[log(p(y|f(x))) q(f(x))] df(x)
        expectation = gauss_hermite_quadrature(log_prob, mean, variance, y=y)

        return expectation


@dataclass
class CollapsedVI(AbstractVariationalInference):
    """Collapsed variational inference for a sparse Gaussian process regression model. 
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables in Sparse Gaussian Processes."""

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood
        self.num_inducing = self.variational_family.num_inducing
        
        if not isinstance(self.likelihood, Gaussian):
            raise TypeError("Likelihood must be Gaussian.")
            
        if not isinstance(self.variational_family, CollapsedVariationalGaussian):
            raise TypeError("Variational family must be CollapsedVariationalGaussian.")

    def elbo(
        self, train_data: Dataset, transformations: Dict, negative: bool = False
    ) -> Callable[[Array], Array]:
        """Compute the evidence lower bound under this model. In short, this requires evaluating the expectation of the model's log-likelihood under the variational approximation. To this, we sum the KL divergence from the variational posterior to the prior. When batching occurs, the result is scaled by the batch size relative to the full dataset size.

        Args:
            train_data (Dataset): The training data for which we should maximise the ELBO with respect to.
            transformations (Dict): The transformation set that unconstrains each parameter.
            negative (bool, optional): Whether or not the resultant elbo function should be negative. For gradient descent where we minimise our objective function this argument should be true as minimisation of the negative corresponds to maximisation of the ELBO. Defaults to False.

        Returns:
            Callable[[Dict, Dataset], Array]: A callable function that accepts a current parameter estimate for which gradients should be computed.
        """
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)
        
        x, y, n = train_data.X, train_data.y, train_data.n
        
        m = self.num_inducing
       

        def elbo_fn(params: Dict) -> Array:
            params = transform(params, transformations)
            noise = params["likelihood"]["obs_noise"]
            z = params["variational_family"]["inducing_inputs"]
            Kzz = gram(self.prior.kernel, z, params["kernel"])
            Kzz += I(m) * self.variational_family.jitter
            Kzx = cross_covariance(self.prior.kernel, z, x, params["kernel"])
            Kxx_diag = diagonal(self.prior.kernel, x, params["kernel"])
            μx = self.prior.mean_function(x, params["mean_function"])
            
            Lz = jnp.linalg.cholesky(Kzz)

            # A = Lz⁻¹ Kzt / σ
            A = jsp.linalg.solve_triangular(Lz, Kzx, lower=True) / jnp.sqrt(noise)

            # AAᵀ
            AAT = jnp.matmul(A, A.T)

            # B = I + AAᵀ
            B = I(m) + AAT

            # LLᵀ = I + AAᵀ
            L = jnp.linalg.cholesky(B)

            # log|B| = 2 trace(log|L|) = 2 Σᵢ log Lᵢᵢ  [since |B| = |LLᵀ| = |L|²  => log|B| = 2 log|L|, and |L| = Πᵢ Lᵢᵢ]
            log_det_B = 2. * jnp.sum(jnp.log(jnp.diagonal(L)))

            diff =  y - μx

            # L⁻¹ A (y - μx)
            L_inv_A_diff = jsp.linalg.solve_triangular(L, jnp.matmul(A, diff), lower=True)

            #  - 1/2 (y - μx)ᵀ (Bσ²)⁻¹ (y - μx) 
            quad = (jnp.sum(L_inv_A_diff ** 2) - jnp.sum((y - μx) ** 2)) / (2. * noise)
            
            # log N(y; μx, Bσ²) = -nπ - n/2 log(σ²) - 1/2 log|B| - 1/2 (y - μx)ᵀ (Bσ²)⁻¹ (y - μx)
            log_prob = - n / 2. * (jnp.log(2. * jnp.pi) + jnp.log(noise)) - log_det_B/2. + quad

            # 1/2 trace(Kxx - AAᵀ)
            trace = (jnp.sum(Kxx_diag) / noise - jnp.trace(AAT)) / 2   
            
            # log N(y; μx, Bσ²) - 1/2 trace(Kxx - AAᵀ)
            return constant * (log_prob - trace).squeeze()

        return elbo_fn