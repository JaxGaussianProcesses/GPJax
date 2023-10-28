# from .classification import GPJaxClassifier
from .config import DefaultConfig
from .regression import GPJaxRegressor
from .scores import (
    LogPredictiveDensity,
    SKLearnScore,
)
from .optim import GPJaxOptimizer, GPJaxOptimiser

__all__ = [
    "GPJaxRegressor",
    # "GPJaxClassifier",
    "LogPredictiveDensity",
    "SKLearnScore",
    "DefaultConfig",
    "GPJaxOptimizer",
    "GPJaxOptimiser",
]
