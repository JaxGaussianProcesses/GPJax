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

from typing import Optional, Tuple

import jax
import jax.random as jr
import optax as ox

from jax.random import KeyArray
from jax._src.random import _check_prng_key
from jaxtyping import Array, Float
from typing import Any

from .base import Module
from .dataset import Dataset
from .objectives import AbstractObjective
from .scan import vscan


def fit(
    *,
    model: Module,
    objective: AbstractObjective,
    train_data: Dataset,
    optim: ox.GradientTransformation,
    num_iters: Optional[int] = 100,
    batch_size: Optional[int] = -1,
    key: Optional[KeyArray] = jr.PRNGKey(42),
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
    unroll: int = 1,
) -> Tuple[Module, Array]:
    """Train a Module model with respect to a supplied Objective function. Optimisers used here should originate from Optax.

    Example:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import optax as ox
        >>> import gpjax as gpx
        >>>
        >>> # (1) Create a dataset:
        >>> X = jnp.linspace(0.0, 10.0, 100)[:, None]
        >>> y = 2.0 * X + 1.0 + 10 * jr.normal(jr.PRNGKey(0), X.shape)
        >>> D = gpx.Dataset(X, y)
        >>>
        >>> # (2) Define your model:
        >>> class LinearModel(gpx.Module):
        ...     weight: float = gpx.param_field()
        ...     bias: float = gpx.param_field()
        ...
        ...     def __call__(self, x):
        ...         return self.weight * x + self.bias
        ...
        >>> model = LinearModel(weight=1.0, bias=1.0)
        >>>
        >>> # (3) Define your loss function:
        >>> class MeanSqaureError(gpx.AbstractObjective):
        ...     def evaluate(self, model: LinearModel, train_data: gpx.Dataset) -> float:
        ...         return jnp.mean((train_data.y - model(train_data.X)) ** 2)
        ...
        >>> loss = MeanSqaureError()
        >>>
        >>> # (4) Train!
        >>> trained_model, history = gpx.fit(
        ...     model=model, objective=loss, train_data=D, optim=ox.sgd(0.001), num_iters=1000
        ... )

    Args:
        model (Module): The model Module to be optimised.
        objective (Objective): The objective function that we are optimising with respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults to 100.
        batch_size (Optional[int]): The size of the mini-batch to use. Defaults to -1 (i.e. full batch).
        key (Optional[KeyArray]): The random key to use for the optimisation batch selection. Defaults to jr.PRNGKey(42).
        log_rate (Optional[int]): How frequently the objective function's value should be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults to True.
        unroll (int): The number of unrolled steps to use for the optimisation. Defaults to 1.

    Returns:
        Tuple[Module, Array]: A Tuple comprising the optimised model and training history respectively.
    """

    # Check inputs.
    _check_model(model)
    _check_objective(objective)
    _check_train_data(train_data)
    _check_optim(optim)
    _check_num_iters(num_iters)
    _check_batch_size(batch_size)
    _check_prng_key(key)
    _check_log_rate(log_rate)
    _check_verbose(verbose)

    # Unconstrained space loss function with stop-gradient rule for non-trainable params.
    def loss(model: Module, batch: Dataset) -> Float[Array, "1"]:
        model = model.stop_gradient()
        return objective(model.constrain(), batch)

    # Unconstrained space model.
    model = model.unconstrain()

    # Initialise optimiser state.
    state = optim.init(model)

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters)

    # Optimisation step.
    def step(carry, key):
        model, opt_state = carry

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, key)
        else:
            batch = train_data

        loss_val, loss_gradient = jax.value_and_grad(loss)(model, batch)
        updates, opt_state = optim.update(loss_gradient, opt_state, model)
        model = ox.apply_updates(model, updates)

        carry = model, opt_state
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan

    # Optimisation loop.
    (model, _), history = scan(step, (model, state), (iter_keys), unroll=unroll)

    # Constrained space.
    model = model.constrain()

    return model, history


def get_batch(train_data: Dataset, batch_size: int, key: KeyArray) -> Dataset:
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.
        key (KeyArray): The random key to use for the batch selection.

    Returns:
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    # Subsample mini-batch indicies with replacement.
    indicies = jr.choice(key, n, (batch_size,), replace=True)

    return Dataset(X=x[indicies], y=y[indicies])


def _check_model(model: Any) -> None:
    """Check that the model is of type Module. Check trainables and bijectors tree structure."""
    if not isinstance(model, Module):
        raise TypeError("model must be of type jaxutils.Module")


def _check_objective(objective: Any) -> None:
    """Check that the objective is of type Objective."""
    if not isinstance(objective, AbstractObjective):
        raise TypeError(f"objective of type {type(objective)} must be of type jaxutils.Objective.")


def _check_train_data(train_data: Any) -> None:
    """Check that the train_data is of type Dataset."""
    if not isinstance(train_data, Dataset):
        raise TypeError("train_data must be of type jaxutils.Dataset")


def _check_optim(optim: Any) -> None:
    """Check that the optimiser is of type GradientTransformation."""
    if not isinstance(optim, ox.GradientTransformation):
        raise TypeError("optax_optim must be of type optax.GradientTransformation")


def _check_num_iters(num_iters: Any) -> None:
    """Check that the number of iterations is of type int and positive."""
    if not isinstance(num_iters, int):
        raise TypeError("num_iters must be of type int")

    if not num_iters > 0:
        raise ValueError("num_iters must be positive")


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

    if not batch_size == -1:
        if not batch_size > 0:
            raise ValueError("batch_size must be positive")


__all__ = [
    "fit",
    "get_batch",
]