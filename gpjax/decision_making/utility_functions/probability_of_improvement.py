# Copyright 2024 The JaxGaussianProcesses Contributors. All Rights Reserved.
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
from dataclasses import dataclass

from beartype.typing import Mapping
from jaxtyping import Num
import tensorflow_probability.substrates.jax as tfp

from gpjax.dataset import Dataset
from gpjax.decision_making.utility_functions.base import (
    AbstractSinglePointUtilityFunctionBuilder,
    SinglePointUtilityFunction,
)
from gpjax.decision_making.utils import (
    OBJECTIVE,
    get_best_latent_observation_val,
)
from gpjax.gps import ConjugatePosterior
from gpjax.typing import (
    Array,
    KeyArray,
)


@dataclass
class ProbabilityOfImprovement(AbstractSinglePointUtilityFunctionBuilder):
    r"""
    An acquisition function which returns the probability of improvement
    of the objective function over the best observed value.

    More precisely, given a predictive posterior distribution of the objective
    function $`f`$, the probability of improvement at a test point $`x`$ is defined as:
    $$`\text{PI}(x) = \text{Prob}[f(x) < f(x_{\text{best}})]`$$
    where $`x_{\text{best}}`$ is the minimiser of the posterior mean
    at previously observed values (to handle noisy observations).

    The probability of improvement can be easily computed using the
    cumulative distribution function of the standard normal distribution $`\Phi`$:
    $$`\text{PI}(x) = \Phi\left(\frac{f(x_{\text{best}}) - \mu}{\sigma}\right)`$$
    where $`\mu`$ and $`\sigma`$ are the mean and standard deviation of the
    predictive distribution of the objective function at $`x`$.

    References
    ----------
    [1] Kushner, H. J. (1964).
    A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise.
    Journal of Basic Engineering, 86(1), 97-106.

    [2] Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016).
    Taking the human out of the loop: A review of Bayesian optimization.
    Proceedings of the IEEE, 104(1), 148-175. doi: 10.1109/JPROC.2015.2494218
    """

    def build_utility_function(
        self,
        posteriors: Mapping[str, ConjugatePosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> SinglePointUtilityFunction:
        """
        Constructs the probability of improvement utility function
        using the predictive posterior of the objective function.

        Args:
            posteriors (Mapping[str, AbstractPosterior]): Dictionary of posteriors to be
            used to form the utility function. One of the posteriors must correspond
            to the `OBJECTIVE` key, as we sample from the objective posterior to form
            the utility function.
            datasets (Mapping[str, Dataset]): Dictionary of datasets which may be used
            to form the utility function. Keys in `datasets` should correspond to
            keys in `posteriors`. One of the datasets must correspond
            to the `OBJECTIVE` key.
            key (KeyArray): JAX PRNG key used for random number generation. Since
            the probability of improvement is computed deterministically
            from the predictive posterior, the key is not used.

        Returns:
            SinglePointUtilityFunction: the probability of improvement utility function.
        """
        self.check_objective_present(posteriors, datasets)

        objective_posterior = posteriors[OBJECTIVE]
        if not isinstance(objective_posterior, ConjugatePosterior):
            raise ValueError(
                "Objective posterior must be a ConjugatePosterior to compute the Probability of Improvement using a Gaussian CDF."
            )

        objective_dataset = datasets[OBJECTIVE]
        if (
            objective_dataset.X is None
            or objective_dataset.n == 0
            or objective_dataset.y is None
        ):
            raise ValueError(
                "Objective dataset must be non-empty to compute the "
                "Probability of Improvement (since we need a "
                "`best_y` value)."
            )

        def probability_of_improvement(x_test: Num[Array, "N D"]):
            best_y = get_best_latent_observation_val(
                objective_posterior, objective_dataset
            )
            predictive_dist = objective_posterior.predict(x_test, objective_dataset)

            normal_dist = tfp.distributions.Normal(
                loc=predictive_dist.mean(),
                scale=predictive_dist.stddev(),
            )

            return normal_dist.cdf(best_y).reshape(-1, 1)

        return probability_of_improvement
