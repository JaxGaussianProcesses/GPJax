# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from warnings import filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

from gpjax import (
    decision_making,
    gps,
    integrators,
    kernels,
    likelihoods,
    mean_functions,
    objectives,
    parameters,
    variational_families,
)
from gpjax.citation import cite
from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.fit import (
    fit,
    fit_scipy,
)
from gpjax.gps import (
    AbstractPosterior,
    AbstractPrior,
    ConjugatePosterior,
    NonConjugatePosterior,
    Prior,
    construct_posterior,
)
from gpjax.integrators import (
    AbstractIntegrator,
    AnalyticalGaussianIntegrator,
    GHQuadratureIntegrator,
    HeteroscedasticGaussianIntegrator,
)
from gpjax.kernels import (
    AbstractKernel,
    Matern12,
    Matern32,
    Matern52,
    RBF,
    RFF,
    Polynomial,
    ProductKernel,
    SumKernel,
)
from gpjax.likelihoods import (
    AbstractLikelihood,
    Bernoulli,
    Gaussian,
    HeteroscedasticGaussian,
    NonGaussian,
    Poisson,
)
from gpjax.mean_functions import (
    AbstractMeanFunction,
    Constant,
    Linear,
    Zero,
)
from gpjax.objectives import (
    AbstractObjective,
    ConjugateLikelihoodObjective,
    ConjugatePredictiveLogLikelihood,
    ConjugateTraceObjective,
    ElboObjective,
    LogLikelihood,
    LogMarginalLikelihood,
    LogPosterior,
    LogPrior,
    MeanFieldElboObjective,
    NegativeLogLikelihood,
    NegativeLogMarginalLikelihood,
    NegativeLogPosterior,
    NegativeLogPrior,
    NegativeMeanFieldElboObjective,
    PredictiveLogLikelihood,
    TraceObjective,
)
from gpjax.parameters import (
    Parameter,
    PositiveReal,
    Real,
    Static,
)
from gpjax.scan import scan
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)
from gpjax.variational_families import (
    AbstractVariationalFamily,
    MeanFieldGaussian,
    VariationalGaussian,
)

import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import Float


def create_heteroscedastic_gp(
    signal_prior: Prior,
    noise_prior: Prior,
    num_datapoints: int,
    latent_g: tp.Union[Float[Array, "N 1"], Parameter, None] = None,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
) -> tp.Tuple[Prior, HeteroscedasticGaussian]:
    """Create a heteroscedastic Gaussian process model.

    This helper function creates a heteroscedastic GP model with separate priors
    for the signal and noise processes, along with a heteroscedastic Gaussian likelihood.

    Args:
        signal_prior (Prior): The GP prior for the signal function f(x).
        noise_prior (Prior): The GP prior for the log noise variance function g(x).
        num_datapoints (int): The number of data points.
        latent_g (Union[Float[Array, "N 1"], Parameter, None]): Initial values for the
            latent log noise variance. If None, initialized to zeros.
        clip_min (float): Minimum value to clip the log noise variance.
        clip_max (float): Maximum value to clip the log noise variance.

    Returns:
        Tuple[Prior, HeteroscedasticGaussian]: The signal prior and heteroscedastic likelihood.
    """
    # Create the heteroscedastic Gaussian likelihood
    likelihood = HeteroscedasticGaussian(
        num_datapoints=num_datapoints,
        latent_g=latent_g,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    return signal_prior, likelihood


__license__ = "MIT"
__description__ = "Didactic Gaussian processes in JAX"
__url__ = "https://github.com/JaxGaussianProcesses/GPJax"
__contributors__ = "https://github.com/JaxGaussianProcesses/GPJax/graphs/contributors"
__version__ = "0.9.4"

__all__ = [
    "AbstractIntegrator",
    "AbstractKernel",
    "AbstractLikelihood",
    "AbstractMeanFunction",
    "AbstractObjective",
    "AbstractPosterior",
    "AbstractPrior",
    "AbstractVariationalFamily",
    "AnalyticalGaussianIntegrator",
    "Array",
    "Bernoulli",
    "cite",
    "ConjugateLikelihoodObjective",
    "ConjugatePosterior",
    "ConjugatePredictiveLogLikelihood",
    "ConjugateTraceObjective",
    "Constant",
    "construct_posterior",
    "create_heteroscedastic_gp",
    "Dataset",
    "ElboObjective",
    "fit",
    "fit_scipy",
    "Gaussian",
    "GaussianDistribution",
    "GHQuadratureIntegrator",
    "HeteroscedasticGaussian",
    "HeteroscedasticGaussianIntegrator",
    "KeyArray",
    "Linear",
    "LogLikelihood",
    "LogMarginalLikelihood",
    "LogPosterior",
    "LogPrior",
    "Matern12",
    "Matern32",
    "Matern52",
    "MeanFieldElboObjective",
    "MeanFieldGaussian",
    "NegativeLogLikelihood",
    "NegativeLogMarginalLikelihood",
    "NegativeLogPosterior",
    "NegativeLogPrior",
    "NegativeMeanFieldElboObjective",
    "NonConjugatePosterior",
    "NonGaussian",
    "Parameter",
    "Poisson",
    "Polynomial",
    "PositiveReal",
    "PredictiveLogLikelihood",
    "Prior",
    "ProductKernel",
    "RBF",
    "Real",
    "RFF",
    "scan",
    "ScalarFloat",
    "Static",
    "SumKernel",
    "TraceObjective",
    "VariationalGaussian",
    "Zero",
]
