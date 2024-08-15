# Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
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
from beartype.typing import (
    Callable,
    Dict,
    Final,
)
import jax.numpy as jnp

from gpjax.dataset import Dataset
from gpjax.gps import AbstractPosterior
from gpjax.typing import (
    Array,
    Float,
)

OBJECTIVE: Final[str] = "OBJECTIVE"
"""
Tag for the objective dataset/function in standard utility functions.
"""


FunctionEvaluator = Callable[[Float[Array, "N D"]], Dict[str, Dataset]]
"""
Type alias for function evaluators, which take an array of points of shape $[N, D]$
and evaluate a set of functions at each point, returning a mapping from function tags
to datasets of the evaluated points. This is the same as the `Observer` in Trieste:
https://github.com/secondmind-labs/trieste/blob/develop/trieste/observer.py
"""


def build_function_evaluator(
    functions: Dict[str, Callable[[Float[Array, "N D"]], Float[Array, "N 1"]]],
) -> FunctionEvaluator:
    """
    Takes a dictionary of functions and returns a `FunctionEvaluator` which can be
    used to evaluate each of the functions at a supplied set of points and return a
    dictionary of datasets storing the evaluated points.
    """
    return lambda x: {tag: Dataset(x, f(x)) for tag, f in functions.items()}


def get_best_latent_observation_val(
    posterior: AbstractPosterior, dataset: Dataset
) -> Float[Array, ""]:
    """
    Takes a posterior and dataset and returns the best (latent) function value in the
    dataset, corresponding to the minimum of the posterior mean value evaluated at
    locations in the dataset. In the noiseless case, this corresponds to the minimum
    value in the dataset.
    """
    return jnp.min(posterior(dataset.X, dataset).mean())
