from dataclasses import dataclass

import jax
from jaxtyping import Num
from loguru import logger
import optax as ox

from gpjax.dataset import Dataset
from gpjax.fit import fit
from gpjax.gps import ConjugatePosterior
from gpjax.kernels import AbstractKernel
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.sklearn.base import BaseEstimator
from gpjax.typing import (
    Array,
    KeyArray,
)
from gpjax.variational_families import CollapsedVariationalGaussian
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


@dataclass
class GPJaxRegressor(BaseEstimator):
    kernel: AbstractKernel
    mean_function: AbstractMeanFunction = None
    n_inducing: int = -1

    def fit(
        self,
        X: Num[Array, "N D"],
        y: Num[Array, "N 1"],
        key: KeyArray,
        compile=True,
        **optim_kwargs,
    ) -> None:
        self.training_data = Dataset(X, y)
        logger.info(f"{self.training_data.n} training data points supplied.")
        self.posterior = self.posterior_builder(self.training_data.n)
        objective, model = self._resolve_inference_scheme(
            self.training_data, self.posterior
        )
        if compile:
            logger.info("Compiling objective function")
            objective = jax.jit(objective)

        if "optim" not in optim_kwargs:
            logger.info("No optimizer specified.")
            logger.info("Using Adam optimizer with learning rate 0.01")
            optim = ox.adam(learning_rate=0.01)
            optim_kwargs["optim"] = optim

        optim_posterior, history = fit(
            model=model,
            objective=objective,
            train_data=self.training_data,
            key=key,
            **optim_kwargs,
        )
        logger.info(f"Starting loss: {history[0]: .3f}")
        logger.info(f"Final loss: {history[-1]: .3f}")
        self.optim_posterior = optim_posterior

    def predict(self, X: Num[Array, "N D"]) -> tfd.MultivariateNormalFullCovariance:
        logger.info(f"Predicting {X.shape[0]} test data points")
        if isinstance(self.optim_posterior, ConjugatePosterior):
            latent_dist = self.optim_posterior.predict(X, train_data=self.training_data)
            predictive_dist = self.optim_posterior.likelihood(latent_dist)
        elif isinstance(self.optim_posterior, CollapsedVariationalGaussian):
            latent_dist = self.optim_posterior.posterior.predict(
                X, train_data=self.training_data
            )
            predictive_dist = self.optim_posterior.posterior.likelihood(latent_dist)
        return predictive_dist

    def posterior_builder(self, n: int):
        return self.prior * Gaussian(num_datapoints=n)
