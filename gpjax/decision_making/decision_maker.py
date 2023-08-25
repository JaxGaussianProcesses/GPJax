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
from dataclasses import (
    dataclass,
    field,
)

from beartype.typing import (
    Callable,
    Dict,
    List,
    Mapping,
)
import jax.random as jr

from gpjax.dataset import Dataset
from gpjax.decision_making.acquisition_functions import (
    AbstractAcquisitionFunctionBuilder,
)
from gpjax.decision_making.acquisition_maximizer import AbstractAcquisitionMaximizer
from gpjax.decision_making.posterior_handler import PosteriorHandler
from gpjax.decision_making.search_space import AbstractSearchSpace
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
    AbstractDecisionMaker abstract base class which handles the core decision loop. The
    decision making loop is split into two key steps, `ask` and `tell`. The `ask`
    step is typically used to decide which point to query next. The `tell` step is
    typically used to update models and datasets with newly queried points.

    Attributes:
        search_space (AbstractSearchSpace): Search space which is being queried
        posterior_handlers (Dict[str, PosteriorHandler]): Dictionary of posterior
            handlers, which are used to update posteriors throughout the decision making
            loop. Tags are used to distinguish between posteriors. In a typical Bayesian
            optimisation setup one of the tags will be `OBJECTIVE`, defined in
            decision_making.utils.
        datasets (Dict[str, Dataset]): Dictionary of datasets, which are augmented with
            observations throughout the decision making loop. In a typical setup they are
            also used to fit the posteriors, using the `posterior_handlers`. Tags are used
            to distinguish datasets, and correspond to tags in `posterior_handlers`.
        acquisition_function_builder (AbstractAcquisitionFunctionBuilder): Object which
            builds acquisition functions from posteriors and datasets, to decide where
            to query next. In a typical Bayesian optimisation setup the point chosen to
            be queried next is the point which maximizes the acquisition function.
        acquisition_maximizer (AbstractAcquisitionMaximizer): Object which maximizes
            acquisition functions over the search space.
        key (KeyArray): JAX random key, used to generate random numbers.
        post_ask (List[Callable]): List of functions to be executed after each ask step.
        post_tell (List[Callable]): List of functions to be executed after each tell
            step.
    """

    search_space: AbstractSearchSpace
    posterior_handlers: Dict[str, PosteriorHandler]
    datasets: Dict[str, Dataset]
    acquisition_function_builder: AbstractAcquisitionFunctionBuilder
    acquisition_maximizer: AbstractAcquisitionMaximizer
    key: KeyArray
    post_ask: List[Callable] = field(
        default_factory=list
    )  # Specific type is List[Callable[[DecisionMaker, Float[Array, ["1 D"]]], None]] but causes Beartype issues
    post_tell: List[Callable] = field(
        default_factory=list
    )  # Specific type is List[Callable[[DecisionMaker], None]] but causes Beartype issues

    @abstractmethod
    def ask(self, key: KeyArray) -> Float[Array, "1 D"]:
        """
        In a typical decision making setup this will use the
        `acquisition_function_builder` to form an acquisition function and then return
        the point which maximizes the acquisition function using the
        `acquisition_maximizer` as the point to be queried next.

        Args:
            key (KeyArray): JAX PRNG key for controlling random state.

        Returns:
            Float[Array, "1 D"]: Point to be queried next
        """
        raise NotImplementedError

    @abstractmethod
    def tell(self, observation_datasets: Mapping[str, Dataset], key: KeyArray):
        """
        Tell decision maker about new observations. In a typical decision making setup
        we will update the datasets and posteriors with the new observations.

        Args:
            observation_datasets (Mapping[str, Dataset]): Dictionary of datasets
                containing new observations. Tags are used to distinguish datasets, and
                correspond to tags in `posterior_handlers` in a typical setup.
            key (KeyArray): JAX PRNG key for controlling random state.
        """
        raise NotImplementedError


@dataclass
class DecisionMaker(AbstractDecisionMaker):
    """
    DecisionMaker class which handles the core decision making loop in a typical setup. The
    decision making loop is split into two key steps, `ask` and `tell`. The `ask`
    step forms an `AcquisitionFunction` from the current `posteriors` and `datasets` and
    returns the point which maximises it. It also stores the formed acquisition function
    under the attribute `self.current_acquisition_function` so that it can be called,
    for instance for plotting, after the `ask` function has been called. The `tell` step
    adds a newly queried point to the `datasets` and updates the `posteriors`.

    This can be run as a typical ask-tell loop, or the `run` method can be used to run
    the decision making loop for a fixed number of steps. Moreover, the `run` method executes
    the functions in `post_ask` and `post_tell` after each ask and tell step
    respectively. This enables the user to add custom functionality, such as the ability
    to plot values of interest during the optimization process.
    """

    def __post_init__(self):
        """
        At initialisation we check that the posterior handlers and datasets are
        consistent (i.e. have the same tags), and then initialise the posteriors, optimizing them using the
        corresponding datasets.
        """
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

    def ask(self, key: KeyArray) -> Float[Array, "1 D"]:
        """
        Get updated acquisition function and return the point which maximises it. This
        method also stores the acquisition function in
        `self.current_acquisition_function` so that it can be accessed after the ask
        function has been called. This is useful for non-deterministic acquisition
        functions, which will differ between calls to `ask` due to the splitting of
        `self.key`.

        Args:
            key (KeyArray): JAX PRNG key for controlling random state.

        Returns:
            Float[Array, "1 D"]: Point to be queried next.
        """
        self.current_acquisition_function = (
            self.acquisition_function_builder.build_acquisition_function(
                self.posteriors, self.datasets, key
            )
        )

        key, _ = jr.split(key)
        return self.acquisition_maximizer.maximize(
            self.current_acquisition_function, self.search_space, key
        )

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
