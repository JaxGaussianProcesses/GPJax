# Copyright 2023 The GPJax Contributors. All Rights Reserved.
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
from abc import (
    ABC,
    abstractmethod,
)
import copy
from dataclasses import dataclass

from beartype.typing import (
    Callable,
    Dict,
    List,
    Mapping,
)
import jax.numpy as jnp
import jax.random as jr

from gpjax.dataset import Dataset
from gpjax.decision_making.posterior_handler import PosteriorHandler
from gpjax.decision_making.search_space import AbstractSearchSpace
from gpjax.decision_making.utility_functions import (
    AbstractUtilityFunctionBuilder,
    ThompsonSampling,
)
from gpjax.decision_making.utility_maximizer import AbstractUtilityMaximizer
from gpjax.decision_making.utils import FunctionEvaluator
from gpjax.gps import AbstractPosterior
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
)


@dataclass
class AbstractDecisionMaker(ABC):
    """
    AbstractDecisionMaker abstract base class which handles the core decision making
    loop, where we sequentially decide on points to query our function of interest at.
    The decision making loop is split into two key steps, `ask` and `tell`. The `ask`
    step is typically used to decide which point to query next. The `tell` step is
    typically used to update models and datasets with newly queried points. These steps
    can be combined in a 'run' loop which alternates between asking which point to query
    next and telling the decision maker about the newly queried point having evaluated
    the black-box function of interest at this point.

    Attributes:
        search_space (AbstractSearchSpace): Search space over which we can evaluate the
        function(s) of interest.
        posterior_handlers (Dict[str, PosteriorHandler]): Dictionary of posterior
            handlers, which are used to update posteriors throughout the decision making
            loop. Note that the word `posteriors` is used for consistency with GPJax, but these
            objects are typically referred to as `models` in the model-based decision
            making literature. Tags are used to distinguish between posteriors. In a typical
            Bayesian optimisation setup one of the tags will be `OBJECTIVE`, defined in
            decision_making.utils.
        datasets (Dict[str, Dataset]): Dictionary of datasets, which are augmented with
            observations throughout the decision making loop. In a typical setup they are
            also used to update the posteriors, using the `posterior_handlers`. Tags are used
            to distinguish datasets, and correspond to tags in `posterior_handlers`.
        key (KeyArray): JAX random key, used to generate random numbers.
        batch_size (int): Number of points to query at each step of the decision making
            loop. Note that `SinglePointUtilityFunction`s are only capable of generating
            one point to be queried at each iteration of the decision making loop.
        post_ask (List[Callable]): List of functions to be executed after each ask step.
        post_tell (List[Callable]): List of functions to be executed after each tell
            step.
    """

    search_space: AbstractSearchSpace
    posterior_handlers: Dict[str, PosteriorHandler]
    datasets: Dict[str, Dataset]
    key: KeyArray
    batch_size: int
    post_ask: List[
        Callable
    ]  # Specific type is List[Callable[[AbstractDecisionMaker, Float[Array, ["B D"]]], None]] but causes Beartype issues
    post_tell: List[
        Callable
    ]  # Specific type is List[Callable[[AbstractDecisionMaker], None]] but causes Beartype issues

    def __post_init__(self):
        """
        At initialisation we check that the posterior handlers and datasets are
        consistent (i.e. have the same tags), and then initialise the posteriors, optimizing them using the
        corresponding datasets.
        """
        self.datasets = copy.copy(
            self.datasets
        )  # Ensure initial datasets passed in to DecisionMaker are not mutated from within

        if self.batch_size < 1:
            raise ValueError(
                f"Batch size must be greater than 0, got {self.batch_size}."
            )

        # Check that posterior handlers and datasets are consistent
        if self.posterior_handlers.keys() != self.datasets.keys():
            raise ValueError(
                "Posterior handlers and datasets must have the same keys. "
                f"Got posterior handlers keys {self.posterior_handlers.keys()} and "
                f"datasets keys {self.datasets.keys()}."
            )

        # Initialize posteriors
        self.posteriors: Dict[str, AbstractPosterior] = {}
        for tag, posterior_handler in self.posterior_handlers.items():
            self.posteriors[tag] = posterior_handler.get_posterior(
                self.datasets[tag], optimize=True, key=self.key
            )

    @abstractmethod
    def ask(self, key: KeyArray) -> Float[Array, "B D"]:
        """
        Get the point(s) to be queried next.

        Args:
            key (KeyArray): JAX PRNG key for controlling random state.

        Returns:
            Float[Array, "1 D"]: Point to be queried next
        """
        raise NotImplementedError

    def tell(self, observation_datasets: Mapping[str, Dataset], key: KeyArray):
        """
        Add newly observed data to datasets and update the corresponding posteriors.

        Args:
            observation_datasets (Mapping[str, Dataset]): Dictionary of datasets
            containing new observations. Tags are used to distinguish datasets, and
            correspond to tags in `posterior_handlers` and `self.datasets`.
            key (KeyArray): JAX PRNG key for controlling random state.
        """
        if observation_datasets.keys() != self.datasets.keys():
            raise ValueError(
                "Observation datasets and existing datasets must have the same keys. "
                f"Got observation datasets keys {observation_datasets.keys()} and "
                f"existing datasets keys {self.datasets.keys()}."
            )

        for tag, observation_dataset in observation_datasets.items():
            self.datasets[tag] += observation_dataset

        for tag, posterior_handler in self.posterior_handlers.items():
            key, _ = jr.split(key)
            self.posteriors[tag] = posterior_handler.update_posterior(
                self.datasets[tag], self.posteriors[tag], optimize=True, key=key
            )

    def run(
        self, n_steps: int, black_box_function_evaluator: FunctionEvaluator
    ) -> Mapping[str, Dataset]:
        """
        Run the decision making loop continuously for for `n_steps`. This is broken down
        into three main steps:
        1. Call the `ask` method to get the point to be queried next.
        2. Call the `black_box_function_evaluator` to evaluate the black box functions
        of interest at the point chosen to be queried.
        3. Call the `tell` method to update the datasets and posteriors with the newly
        observed data.

        In addition to this, after the `ask` step, the functions in the `post_ask` list
        are executed, taking as arguments the decision maker and the point chosen to be
        queried next. Similarly, after the `tell` step, the functions in the `post_tell`
        list are executed, taking the decision maker as the sole argument.

        Args:
            n_steps (int): Number of steps to run the decision making loop for.
            black_box_function_evaluator (FunctionEvaluator): Function evaluator which
                evaluates the black box functions of interest at supplied points.

        Returns:
            Mapping[str, Dataset]: Dictionary of datasets containing the observations
            made throughout the decision making loop, as well as the initial data
            supplied when initialising the `DecisionMaker`.
        """
        for _ in range(n_steps):
            query_point = self.ask(self.key)

            for post_ask_method in self.post_ask:
                post_ask_method(self, query_point)

            self.key, _ = jr.split(self.key)
            observation_datasets = black_box_function_evaluator(query_point)
            self.tell(observation_datasets, self.key)

            for post_tell_method in self.post_tell:
                post_tell_method(self)

        return self.datasets


@dataclass
class UtilityDrivenDecisionMaker(AbstractDecisionMaker):
    """
    UtilityDrivenDecisionMaker class which handles the core decision making loop in a
    typical model-based decision making setup. In this setup we use surrogate model(s)
    for the function(s) of interest, and define a utility function (often called the
    'acquisition function' in the context of Bayesian optimisation) which characterises
    how useful it would be to query a given point within the search space given the data
    we have observed so far. This can then be used to decide which point(s) to query
    next.

    The decision making loop is split into two key steps, `ask` and `tell`. The `ask`
    step forms a `UtilityFunction` from the current `posteriors` and `datasets` and
    returns the point which maximises it. It also stores the formed utility function
    under the attribute `self.current_utility_function` so that it can be called,
    for instance for plotting, after the `ask` function has been called. The `tell` step
    adds a newly queried point to the `datasets` and updates the `posteriors`.

    This can be run as a typical ask-tell loop, or the `run` method can be used to run
    the decision making loop for a fixed number of steps. Moreover, the `run` method executes
    the functions in `post_ask` and `post_tell` after each ask and tell step
    respectively. This enables the user to add custom functionality, such as the ability
    to plot values of interest during the optimization process.

    Attributes:
        utility_function_builder (AbstractUtilityFunctionBuilder): Object which
                builds utility functions from posteriors and datasets, to decide where
                to query next. In a typical Bayesian optimisation setup the point chosen to
                be queried next is the point which maximizes the utility function.
        utility_maximizer (AbstractUtilityMaximizer): Object which maximizes
            utility functions over the search space.
    """

    utility_function_builder: AbstractUtilityFunctionBuilder
    utility_maximizer: AbstractUtilityMaximizer

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size > 1 and not isinstance(
            self.utility_function_builder, ThompsonSampling
        ):
            raise NotImplementedError(
                "Batch size > 1 currently only supported for Thompson sampling."
            )

    def ask(self, key: KeyArray) -> Float[Array, "B D"]:
        """
        Get updated utility function(s) and return the point(s) which maximises it/them. This
        method also stores the utility function(s) in
        `self.current_utility_functions` so that they can be accessed after the ask
        function has been called. This is useful for non-deterministic utility
        functions, which may differ between calls to `ask` due to the splitting of
        `self.key`.

        Note that in general `SinglePointUtilityFunction`s are only capable of
        generating one point to be queried at each iteration of the decision making loop
        (i.e. `self.batch_size` must be 1). However, Thompson sampling can be used in a
        batched setting by drawing a batch of different samples from the GP posterior.
        This is done by calling `build_utility_function` with different keys
        sequentilly, and optimising each of these individual samples in sequence in
        order to obtain `self.batch_size` points to query next.

        Args:
            key (KeyArray): JAX PRNG key for controlling random state.

        Returns:
            Float[Array, "B D"]: Point(s) to be queried next.
        """
        self.current_utility_functions = []
        maximizers = []
        # We currently only allow Thompson sampling to be run with batch size > 1. More
        # batched utility functions may be added in the future.
        if isinstance(self.utility_function_builder, ThompsonSampling) or (
            (not isinstance(self.utility_function_builder, ThompsonSampling))
            and (self.batch_size == 1)
        ):
            # Draw 'self.batch_size' Thompson samples and optimize each of them in order to
            # obtain 'self.batch_size' points to query next.
            for _ in range(self.batch_size):
                decision_function = (
                    self.utility_function_builder.build_utility_function(
                        self.posteriors, self.datasets, key
                    )
                )
                self.current_utility_functions.append(decision_function)

                _, key = jr.split(key)
                maximizer = self.utility_maximizer.maximize(
                    decision_function, self.search_space, key
                )
                maximizers.append(maximizer)
                _, key = jr.split(key)

            maximizers = jnp.concatenate(maximizers)
            return maximizers
        else:
            raise NotImplementedError(
                "Only Thompson sampling currently supports batch size > 1."
            )
