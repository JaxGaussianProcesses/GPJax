from abc import abstractmethod
from dataclasses import dataclass

import jax
from loguru import logger

from gpjax.dataset import Dataset
from gpjax.gps import (
    AbstractPosterior,
    Prior,
)
from gpjax.kernels import AbstractKernel
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import (
    AbstractMeanFunction,
    Constant,
)
from gpjax.objectives import (
    ELBO,
    CollapsedELBO,
    ConjugateMLL,
    LogPosteriorDensity,
)
from gpjax.sklearn import DefaultConfig
from gpjax.variational_families import (
    CollapsedVariationalGaussian,
    VariationalGaussian,
)
from gpjax.typing import KeyArray


@dataclass
class BaseEstimator:
    kernel: AbstractKernel
    mean_function: AbstractMeanFunction = None
    n_inducing: int = -1
    config: DefaultConfig = None

    def __post_init__(self):
        if self.mean_function is None:
            logger.info("No mean function specified, using constant mean function")
            self.mean_function = Constant()
        logger.info(f"Building GP prior with {self.kernel.name} kernel")
        self.prior = Prior(kernel=self.kernel, mean_function=self.mean_function)
        self.posterior = None
        self.optim_posterior = None
        if self.config is None:
            self.config = DefaultConfig()

    @abstractmethod
    def fit(
        self,
        X: jax.Array,
        y: jax.Array,
        key: KeyArray,
        compile=True,
        **optim_kwargs,
    ) -> None:
        raise NotImplementedError("Please implement `fit` in your subclass")

    @abstractmethod
    def predict(self, X: jax.Array, y: jax.Array) -> None:
        raise NotImplementedError("Please implement `predict` in your subclass")

    def predict_mean(self, X: jax.Array) -> jax.Array:
        predictive_dist = self.predict(X)
        return predictive_dist.mean()

    def predict_stddev(self, X: jax.Array) -> jax.Array:
        predictive_dist = self.predict(X)
        return predictive_dist.stddev()

    def predict_mean_and_stddev(self, X: jax.Array) -> jax.Array:
        predictive_dist = self.predict(X)
        return predictive_dist.mean(), predictive_dist.stddev()

    def score(self, X, y, scoring_fn):
        predictive_dist = self.predict(X)
        score = scoring_fn(predictive_dist, y)
        logger.info(f"{scoring_fn.name}: {score: .3f}")
        return score

    def _resolve_inference_scheme(self, dataset: Dataset, posterior: AbstractPosterior):
        n_data = dataset.n
        conjugacy = isinstance(posterior.likelihood, Gaussian)
        self._set_inducing_points(n_data)
        objective, model = self._get_objective_and_model(
            n_data, conjugacy, dataset, posterior
        )
        return objective, model

    def _set_inducing_points(self, n_data):
        if self.n_inducing == -1:
            logger.info("No inducing points specified, using default")
            self.n_inducing = self.config.n_inducing_heuristic(n_data)
        else:
            logger.info(f"{self.n_inducing} inducing points specified")

    def _get_objective_and_model(self, n_data, conjugacy, dataset, posterior):
        if n_data < self.config.sparse_threshold and self.n_inducing == -1:
            return self._get_objective_and_model_for_small_data(conjugacy, posterior)
        else:
            return self._get_objective_and_model_for_large_data(
                n_data, conjugacy, dataset, posterior
            )

    def _get_objective_and_model_for_small_data(self, conjugacy, posterior):
        model = posterior
        if conjugacy:
            objective = ConjugateMLL(negative=True)
            logger.info("Using conjugate marginal log likelihood")
        else:
            objective = LogPosteriorDensity(negative=True)
            logger.info("Using log posterior density")
        return objective, model

    def _get_objective_and_model_for_large_data(
        self, n_data, conjugacy, dataset, posterior
    ):
        z = self.config.inducing_point_selector(dataset.X, self.n_inducing)
        if conjugacy and n_data < self.config.stochastic_threshold:
            model = CollapsedVariationalGaussian(posterior=posterior, inducing_inputs=z)
            objective = CollapsedELBO(negative=True)
            logger.info("Using collapsed ELBO")
        else:
            model = VariationalGaussian(posterior=posterior, inducing_inputs=z)
            objective = ELBO(negative=True)
            logger.info("Using uncollapsed ELBO")
        return objective, model
