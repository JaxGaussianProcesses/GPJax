from abc import abstractmethod
from dataclasses import dataclass

from beartype.typing import Tuple
from jaxtyping import Num
from loguru import logger
import tensorflow_probability.substrates.jax.distributions as tfd

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
    AbstractObjective,
    CollapsedELBO,
    ConjugateMLL,
    LogPosteriorDensity,
)
from gpjax.sklearn import DefaultConfig
from gpjax.sklearn.scores import AbstractScore
from gpjax.typing import Array, KeyArray, ScalarFloat
from gpjax.variational_families import (
    AbstractVariationalFamily,
    CollapsedVariationalGaussian,
    VariationalGaussian,
)


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
        X: Num[Array, "N D"],
        y: Num[Array, "N 1"],
        key: KeyArray,
        compile=True,
        **optim_kwargs,
    ) -> None:
        raise NotImplementedError("Please implement `fit` in your subclass")

    @abstractmethod
    def predict(self, X: Num[Array, "N D"]) -> tfd.Distribution:
        raise NotImplementedError("Please implement `predict` in your subclass")

    def predict_mean(self, X: Num[Array, "N D"]) -> Num[Array, "N"]:
        predictive_dist = self.predict(X)
        return predictive_dist.mean()

    def predict_stddev(self, X: Num[Array, "N D"]) -> Num[Array, "N"]:
        predictive_dist = self.predict(X)
        return predictive_dist.stddev()

    def predict_mean_and_stddev(
        self, X: Num[Array, "N D"]
    ) -> Tuple[Num[Array, "N"], Num[Array, "N"]]:
        predictive_dist = self.predict(X)
        return predictive_dist.mean(), predictive_dist.stddev()

    def score(
        self, X: Num[Array, "N D"], y: Num[Array, "N D"], scoring_fn: AbstractScore
    ) -> ScalarFloat:
        predictive_dist = self.predict(X)
        score = scoring_fn(predictive_dist, y)
        logger.info(f"{scoring_fn.name}: {score: .3f}")
        return score

    def _resolve_inference_scheme(self, dataset: Dataset, posterior: AbstractPosterior):
        n_data = dataset.n
        conjugacy = isinstance(posterior.likelihood, Gaussian)
        objective, model = self._get_objective_and_model(
            n_data, conjugacy, dataset, posterior
        )
        return objective, model

    def _set_inducing_points(self, n_data: int):
        if self.n_inducing == -1:
            logger.info("No inducing points specified, using default")
            self.n_inducing = self.config.n_inducing_heuristic(n_data)
        else:
            logger.info(f"{self.n_inducing} inducing points specified")

    def _get_objective_and_model(
        self,
        n_data: int,
        conjugacy: bool,
        dataset: Dataset,
        posterior: AbstractPosterior,
    ):
        if n_data < self.config.sparse_threshold and self.n_inducing == -1:
            return self._get_objective_and_model_for_small_data(conjugacy, posterior)
        else:
            return self._get_objective_and_model_for_large_data(
                n_data, conjugacy, dataset, posterior
            )

    def _get_objective_and_model_for_small_data(
        self, conjugacy: bool, posterior: AbstractPosterior
    ) -> Tuple[AbstractObjective, AbstractPosterior]:
        model = posterior
        if conjugacy:
            logger.info("Conjugate likelihood identified")
            objective = ConjugateMLL(negative=True)
            logger.info("Using conjugate marginal log likelihood")
        else:
            logger.info("Non-conjugate likelihood identified")
            objective = LogPosteriorDensity(negative=True)
            logger.info("Using log posterior density")
        return objective, model

    def _get_objective_and_model_for_large_data(
        self,
        n_data: int,
        conjugacy: bool,
        dataset: Dataset,
        posterior: AbstractPosterior,
    ) -> Tuple[AbstractObjective, AbstractVariationalFamily]:
        self._set_inducing_points(n_data)
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
