import abc
from typing import Any, Callable, Dict

import jax.numpy as jnp
from chex import dataclass
from jax import vmap

from .gps import AbstractPosterior
from .parameters import transform
from .quadrature import gauss_hermite_quadrature
from .types import Array, Dataset
from .utils import concat_dictionaries
from .variational import VariationalFamily


@dataclass
class VariationalPosterior:
    """A variational posterior object. With reference to some true posterior distribution :math:`p`, this can be used to minimise the KL-divergence between :math:`p` and a variational posterior :math:`q`."""

    posterior: AbstractPosterior
    variational_family: VariationalFamily

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
class SVGP(VariationalPosterior):
    """The Sparse Variational Gaussian Process (SVGP) variational posterior. The key reference is Hensman et. al., (2013) - Gaussian processes for big data."""

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
            kl = self.variational_family.prior_kl(params)
            var_exp = self.variational_expectation(params, batch)

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

        mean, variance = vmap(self.variational_family.pred_moments, in_axes=(None, 0))(
            params, x[:, jnp.newaxis, :]
        )

        # Get log(p(y|F)) function for current likelihood parameter values:
        def log_prob(F, y):
            return self.likelihood.link_function(F, params["likelihood"]).log_prob(y)

        return gauss_hermite_quadrature(log_prob, mean.squeeze(1), variance.squeeze(1), y=y)