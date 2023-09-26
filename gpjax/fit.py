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
from dataclasses import (
    asdict,
    dataclass,
)

from beartype.typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import jax
from jax._src.random import _check_prng_key
import jax.numpy as jnp
import jax.random as jr
from jaxopt import (
    OptaxSolver,
    ScipyMinimize,
)

from gpjax.base import Module
from gpjax.dataset import Dataset
from gpjax.scan import vscan
from gpjax.typing import (
    Array,
    KeyArray,
)

ModuleModel = TypeVar("ModuleModel", bound=Module)


def fit(  # noqa: PLR0913
    *,
    model: ModuleModel,
    train_data: Dataset,
    solver: Union[ScipyMinimize, OptaxSolver],
    key: KeyArray,
    batch_size: Optional[int] = -1,
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
    unroll: Optional[int] = 1,
    safe: Optional[bool] = True,
) -> Tuple[ModuleModel, Array]:
    r"""Train a Module model with respect to a supplied Objective function.
    `solver` must be an instance of `jaxopt`'s `OptaxSolver` or `ScipyMinimze`.

    Example:
    ```python
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import jaxopt
        >>> import gpjax as gpx
        >>>
        >>> # (1) Create a dataset:
        >>> X = jnp.linspace(0.0, 10.0, 100)[:, None]
        >>> y = 2.0 * X + 1.0 + 10 * jr.normal(jr.PRNGKey(0), X.shape)
        >>> D = gpx.Dataset(X, y)
        >>>
        >>> # (2) Define your model:
        >>> class LinearModel(gpx.Module):
                weight: float = gpx.param_field()
                bias: float = gpx.param_field()

                def __call__(self, x):
                    return self.weight * x + self.bias

        >>> model = LinearModel(weight=1.0, bias=1.0)
        >>>
        >>> # (3) Define your loss function:
        >>> class MeanSquareError(gpx.AbstractObjective):
                def evaluate(self, model: LinearModel, train_data: gpx.Dataset) -> float:
                    return jnp.mean((train_data.y - model(train_data.X)) ** 2)
        >>>
        >>> loss = MeanSquaredError()
        >>>
        >>> # (4) Train!
        >>> trained_model, history = gpx.fit(
                model=model,
                train_data=D,
                solver=jaxopt.ScipyMinimize(fun=loss),
            )
    ```

    Args:
        model (Module): The model Module to be optimised.
        train_data (Dataset): The training data to be used for the optimisation.
        solver (Union[SCipyMinimize, OptaxSolver])): The `jaxopt` solver.
        batch_size (Optional[int]): The size of the mini-batch to use. Defaults to -1
            (i.e. full batch).
        key (Optional[KeyArray]): The random key to use for the optimisation batch
            selection. Defaults to jr.PRNGKey(42).
        log_rate (Optional[int]): How frequently the objective function's value should
            be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults
            to True.
        unroll (int): The number of unrolled steps to use for the optimisation.
            Defaults to 1.

    Returns
    -------
        Tuple[Module, Array]: A Tuple comprising the optimised model and training
            history respectively.
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_batch_size(batch_size)
        _check_prng_key(key)
        _check_log_rate(log_rate)
        _check_verbose(verbose)

    if isinstance(solver, ScipyMinimize) and batch_size != -1:
        raise ValueError("ScipyMinimze optimizers do not support batching")

    # Unconstrained space model.
    model = model.unconstrain()

    # Initialise solver state.
    solver.fun = _wrap_objective(solver.fun)

    if isinstance(solver, OptaxSolver):  # hack for Optax compatibility
        model = jax.tree_map(lambda x: x.astype(jnp.float64), model)
    # # elif isinstance(solver, ScipyMinimize): # hack for jaxopt compatibility
    # del solver.options["maxiter"]

    solver.__post_init__()  # needed to propagate changes to `fun` attribute

    if isinstance(solver, OptaxSolver):  # For optax, run optimization by step
        solver_state = solver.init_state(
            model,
            get_batch(train_data, batch_size, key) if batch_size != -1 else train_data,
        )

        # Mini-batch random keys to scan over.
        iter_keys = jr.split(key, solver.maxiter)

        # Optimisation step.
        def step(carry, key):
            model, state = carry

            if batch_size != -1:
                batch = get_batch(train_data, batch_size, key)
            else:
                batch = train_data

            model, state = solver.update(model, state, batch)
            carry = model, state
            return carry, state.value

        # Optimisation scan.
        scan = vscan if verbose else jax.lax.scan

        # Optimisation loop.
        (model, _), history = scan(
            step, (model, solver_state), (iter_keys), unroll=unroll
        )

    elif isinstance(solver, ScipyMinimize):  # Scipy runs whole optimization loop
        initial_loss = solver.fun(model, train_data)
        model, result = solver.run(model, train_data)
        history = jnp.array([initial_loss, result.fun_val])
        if verbose:
            print(f" Found model with loss {result.fun_val}")

    # Constrained space.
    model = model.constrain()

    return model, history


def get_batch(train_data: Dataset, batch_size: int, key: KeyArray) -> Dataset:
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.
        key (KeyArray): The random key to use for the batch selection.

    Returns
    -------
        Dataset: The batched dataset.
    """
    x, y, n, mask = train_data.X, train_data.y, train_data.n, train_data.mask

    # Subsample mini-batch indices with replacement.
    indices = jr.choice(key, n, (batch_size,), replace=True)

    return Dataset(X=x[indices], y=y[indices], mask=mask[indices] if mask else None)


def _wrap_objective(objective: Callable):
    def wrapped(model, batch):
        model = model.stop_gradient()
        return objective(model.constrain(), batch)

    return wrapped


def _check_model(model: Any) -> None:
    """Check that the model is of type Module. Check trainables and bijectors tree structure."""
    if not isinstance(model, Module):
        raise TypeError("model must be of type gpjax.Module")


def _check_train_data(train_data: Any) -> None:
    """Check that the train_data is of type Dataset."""
    if not isinstance(train_data, Dataset):
        raise TypeError("train_data must be of type gpjax.Dataset")


def _check_log_rate(log_rate: Any) -> None:
    """Check that the log rate is of type int and positive."""
    if not isinstance(log_rate, int):
        raise TypeError("log_rate must be of type int")

    if not log_rate > 0:
        raise ValueError("log_rate must be positive")


def _check_verbose(verbose: Any) -> None:
    """Check that the verbose is of type bool."""
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be of type bool")


def _check_batch_size(batch_size: Any) -> None:
    """Check that the batch size is of type int and positive if not minus 1."""
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be of type int")

    if not batch_size == -1 and not batch_size > 0:
        raise ValueError("batch_size must be positive")


__all__ = [
    "fit",
    "get_batch",
]
