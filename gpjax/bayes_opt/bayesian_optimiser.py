from dataclasses import (
    dataclass,
    field,
)

from beartype.typing import (
    Callable,
    List,
    Mapping,
    Optional,
)
import jax.random as jr

from gpjax.bayes_opt.acquisition_functions import AbstractAcquisitionFunctionBuilder
from gpjax.bayes_opt.acquisition_optimiser import AbstractAcquisitionOptimiser
from gpjax.bayes_opt.function_evaluator import FunctionEvaluator
from gpjax.bayes_opt.posterior_optimiser import AbstractPosteriorOptimiser
from gpjax.bayes_opt.search_space import AbstractSearchSpace
from gpjax.dataset import Dataset
from gpjax.gps import AbstractPosterior
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
)


@dataclass
class BayesianOptimiser:
    """
    BayesianOptimiser class which handles the core Bayesian optimisation loop. The
    Bayesian optimisation loop is split into two key steps, `ask` and `tell`. The `ask`
    step forms an `AcquisitionFunction` from the current `posterioras and `datasets` and
    returns the point which maximises it. The `tell` step adds newly queried point to the
    `datasets` and updates the `posteriors`.

    This can be run as a typical ask-tell loop, or the `run` method can be used to run
    the optimisation for a fixed number of steps. Moreover, the `run` method executes
    the functions in `post_ask` and `post_tell` after each ask and tell step
    respectively. This enables the user to add custom functionality, such as the ability
    to plot values of interest during the optimisation process.
    """

    search_space: AbstractSearchSpace
    posteriors: Mapping[str, AbstractPosterior]
    datasets: Mapping[str, Dataset]
    posterior_optimiser: AbstractPosteriorOptimiser
    acquisition_function_builder: AbstractAcquisitionFunctionBuilder
    acquisition_optimiser: AbstractAcquisitionOptimiser
    key: KeyArray
    fit_initial_posteriors: bool = True
    black_box_function_evaluator: Optional[FunctionEvaluator] = None
    post_ask: List[Callable] = field(
        default_factory=list
    )  # TODO: Make type more specific? Function takes `self` and most recently queried point as input
    post_tell: List[Callable] = field(
        default_factory=list
    )  # TODO: Make type more specific? Function takes `self` as input

    def __post_init__(self):
        if self.fit_initial_posteriors:
            self.posteriors = self.posterior_optimiser.optimise(
                self.posteriors, self.datasets, self.key
            )

    def ask(self) -> Float[Array, "1 D"]:
        """
        Get updated acquisition function and return the point which maximises it
        """

        self.key, subkey = jr.split(self.key)
        acquisition_function = (
            self.acquisition_function_builder.build_acquisition_function(
                self.posteriors, self.datasets, subkey
            )
        )

        self.key, subkey = jr.split(self.key)
        return self.acquisition_optimiser.optimise(
            acquisition_function, self.search_space, subkey
        )

    def tell(self, observation_datasets: Mapping[str, Dataset]):
        """
        Add new data to datasets and update models
        """
        for tag, observation_dataset in observation_datasets.items():
            self.datasets[tag] += observation_dataset

        self.key, subkey = jr.split(self.key)
        self.posteriors = self.posterior_optimiser.optimise(
            self.posteriors, self.datasets, subkey
        )

    def run(self, n_steps: int) -> Mapping[str, Dataset]:
        """
        Run Bayesian optimisation for n_steps
        """
        for i in range(n_steps):
            print(f"Starting Iteration {i}")
            query_point = self.ask()

            for post_ask_method in self.post_ask:
                post_ask_method(self, query_point)

            if self.black_box_function_evaluator is None:
                raise ValueError(
                    "No function evaluator provided, cannot evaluate query point"
                )

            observation_datasets = self.black_box_function_evaluator(query_point)
            self.tell(observation_datasets)

            for post_tell_method in self.post_tell:
                post_tell_method(self)

        return self.datasets
