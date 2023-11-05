# from .classification import GPJaxClassifier
from gpjax.sklearn.config import DefaultConfig
from gpjax.sklearn.optim import (
    GPJaxOptimiser,
    GPJaxOptimizer,
)
from gpjax.sklearn.regression import GPJaxRegressor
from gpjax.sklearn.scores import (
    LogPredictiveDensity,
    SKLearnScore,
)

__all__ = [
    "GPJaxRegressor",
    # "GPJaxClassifier",
    "LogPredictiveDensity",
    "SKLearnScore",
    "DefaultConfig",
    "GPJaxOptimizer",
    "GPJaxOptimiser",
]
