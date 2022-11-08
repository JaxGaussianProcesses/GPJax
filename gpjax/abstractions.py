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

from typing import Callable, Dict, Optional, Tuple, Any, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from chex import dataclass
from jax import lax
from jax.experimental import host_callback
from jaxtyping import Array, Float
from tqdm.auto import tqdm

from .natural_gradients import natural_gradients
from .parameters import ParameterState, constrain, trainable_params, unconstrain
from .types import Dataset, PRNGKeyType
from .variational_inference import StochasticVI


@dataclass(frozen=True)
class InferenceState:
    """Imutable dataclass for storing optimised parameters and training history."""

    params: Dict
    history: Float[Array, "n_iters"]

    def unpack(self) -> Tuple[Dict, Float[Array, "n_iters"]]:
        """Unpack parameters and training history into a tuple.

        Returns:
            Tuple[Dict, Float[Array, "n_iters"]]: Tuple of parameters and training history.
        """
        return self.params, self.history


def fit(
    objective: Callable,
    parameter_state: ParameterState,
    optax_optim: ox.GradientTransformation,
    n_iters: Optional[int] = 100,
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
) -> InferenceState:
    """Abstracted method for fitting a GP model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.

    Args:
        objective (Callable): The objective function that we are optimising with respect to.
        parameter_state (ParameterState): The initial parameter state.
        optax_optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.
        verbose (bool, optional): Whether to print the training loading bar. Defaults to True.

    Returns:
        InferenceState: An InferenceState object comprising the optimised parameters and training history respectively.
    """

    params, trainables, bijectors = parameter_state.unpack()

    # Define optimisation loss function on unconstrained space, with a stop gradient rule for trainables that are set to False
    def loss(params: Dict) -> Float[Array, "1"]:
        params = trainable_params(params, trainables)
        params = constrain(params, bijectors)
        return objective(params)

    # Tranform params to unconstrained space
    params = unconstrain(params, bijectors)

    # Initialise optimiser state
    opt_state = optax_optim.init(params)

    # Iteration loop numbers to scan over
    iter_nums = jnp.arange(n_iters)

    # Optimisation step
    def step(carry, iter_num: int):
        params, opt_state = carry
        loss_val, loss_gradient = jax.value_and_grad(loss)(params)
        updates, opt_state = optax_optim.update(loss_gradient, opt_state, params)
        params = ox.apply_updates(params, updates)
        carry = params, opt_state
        return carry, loss_val

    # Display progress bar if verbose is True
    if verbose:
        step = progress_bar_scan(n_iters, log_rate)(step)

    # Run the optimisation loop
    (params, _), history = jax.lax.scan(step, (params, opt_state), iter_nums)

    # Tranform final params to constrained space
    params = constrain(params, bijectors)

    return InferenceState(params=params, history=history)


def fit_batches(
    objective: Callable,
    parameter_state: ParameterState,
    train_data: Dataset,
    optax_optim: ox.GradientTransformation,
    key: PRNGKeyType,
    batch_size: int,
    n_iters: Optional[int] = 100,
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
) -> InferenceState:
    """Abstracted method for fitting a GP model with mini-batches respect to a
    supplied objective function.
    Optimisers used here should originate from Optax.

    Args:
        objective (Callable): The objective function that we are optimising with respect to.
        parameter_state (ParameterState): The parameters for which we would like to minimise our objective function with.
        train_data (Dataset): The training dataset.
        optax_optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        key (PRNGKeyType): The PRNG key for the mini-batch sampling.
        batch_size(int): The batch_size.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.
        verbose (bool, optional): Whether to print the training loading bar. Defaults to True.

    Returns:
        InferenceState: An InferenceState object comprising the optimised parameters and training history respectively.
    """

    params, trainables, bijectors = parameter_state.unpack()

    # Define optimisation loss function on unconstrained space, with a stop gradient rule for trainables that are set to False
    def loss(params: Dict, batch: Dataset) -> Float[Array, "1"]:
        params = trainable_params(params, trainables)
        params = constrain(params, bijectors)
        return objective(params, batch)

    # Tranform params to unconstrained space
    params = unconstrain(params, bijectors)

    # Initialise optimiser state
    opt_state = optax_optim.init(params)

    # Mini-batch random keys and iteration loop numbers to scan over
    keys = jr.split(key, n_iters)
    iter_nums = jnp.arange(n_iters)

    # Optimisation step
    def step(carry, iter_num__and__key):
        iter_num, key = iter_num__and__key
        params, opt_state = carry

        batch = get_batch(train_data, batch_size, key)

        loss_val, loss_gradient = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optax_optim.update(loss_gradient, opt_state, params)
        params = ox.apply_updates(params, updates)

        carry = params, opt_state
        return carry, loss_val

    # Display progress bar if verbose is True
    if verbose:
        step = progress_bar_scan(n_iters, log_rate)(step)

    # Run the optimisation loop
    (params, _), history = jax.lax.scan(step, (params, opt_state), (iter_nums, keys))

    # Tranform final params to constrained space
    params = constrain(params, bijectors)

    return InferenceState(params=params, history=history)


def get_batch(train_data: Dataset, batch_size: int, key: PRNGKeyType) -> Dataset:
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.

    Returns:
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    # Subsample data inidicies with replacement to get the mini-batch
    indicies = jr.choice(key, n, (batch_size,), replace=True)

    return Dataset(X=x[indicies], y=y[indicies])


def fit_natgrads(
    stochastic_vi: StochasticVI,
    parameter_state: ParameterState,
    train_data: Dataset,
    moment_optim: ox.GradientTransformation,
    hyper_optim: ox.GradientTransformation,
    key: PRNGKeyType,
    batch_size: int,
    n_iters: Optional[int] = 100,
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
) -> Dict:
    """This is a training loop for natural gradients. See Salimbeni et al.
    (2018) Natural Gradients in Practice: Non-Conjugate Variational Inference
    in Gaussian Process Models
    Each iteration comprises a hyperparameter gradient step followed by natural
    gradient step to avoid a stale posterior.

    Args:
        stochastic_vi (StochasticVI): The stochastic variational inference algorithm to be used for training.
        parameter_state (ParameterState): The initial parameter state.
        train_data (Dataset): The training dataset.
        moment_optim (GradientTransformation): The Optax optimiser for the natural gradient updates on the moments.
        hyper_optim (GradientTransformation): The Optax optimiser for gradient updates on the hyperparameters.
        key (PRNGKeyType): The PRNG key for the mini-batch sampling.
        batch_size(int): The batch_size.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.
        verbose (bool, optional): Whether to print the training loading bar. Defaults to True.

    Returns:
        InferenceState: A dataclass comprising optimised parameters and training history.
    """

    params, trainables, bijectors = parameter_state.unpack()

    # Tranform params to unconstrained space
    params = unconstrain(params, bijectors)

    # Initialise optimiser states
    hyper_state = hyper_optim.init(params)
    moment_state = moment_optim.init(params)

    # Build natural and hyperparameter gradient functions
    nat_grads_fn, hyper_grads_fn = natural_gradients(
        stochastic_vi, train_data, bijectors, trainables
    )

    # Mini-batch random keys and iteration loop numbers to scan over
    keys = jax.random.split(key, n_iters)
    iter_nums = jnp.arange(n_iters)

    # Optimisation step
    def step(carry, iter_num__and__key):
        iter_num, key = iter_num__and__key
        params, hyper_state, moment_state = carry

        batch = get_batch(train_data, batch_size, key)

        # Hyper-parameters update:
        loss_val, loss_gradient = hyper_grads_fn(params, batch)
        updates, hyper_state = hyper_optim.update(loss_gradient, hyper_state, params)
        params = ox.apply_updates(params, updates)

        # Natural gradients update:
        loss_val, loss_gradient = nat_grads_fn(params, batch)
        updates, moment_state = moment_optim.update(loss_gradient, moment_state, params)
        params = ox.apply_updates(params, updates)

        carry = params, hyper_state, moment_state
        return carry, loss_val

    # Display progress bar if verbose is True
    if verbose:
        step = progress_bar_scan(n_iters, log_rate)(step)

    # Run the optimisation loop
    (params, _, _), history = jax.lax.scan(
        step, (params, hyper_state, moment_state), (iter_nums, keys)
    )

    # Tranform final params to constrained space
    params = constrain(params, bijectors)

    return InferenceState(params=params, history=history)


def progress_bar_scan(n_iters: int, log_rate: int) -> Callable:
    """Progress bar for Jax.lax scans (adapted from https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/)."""

    tqdm_bars = {}
    remainder = n_iters % log_rate

    def _define_tqdm(args: Any, transform: Any) -> None:
        """Define a tqdm progress bar."""
        tqdm_bars[0] = tqdm(range(n_iters))

    def _update_tqdm(args: Any, transform: Any) -> None:
        """Update the tqdm progress bar with the latest objective value."""
        loss_val, arg = args
        tqdm_bars[0].update(arg)
        tqdm_bars[0].set_postfix({"Objective": f"{loss_val: .2f}"})

    def _close_tqdm(args: Any, transform: Any) -> None:
        """Close the tqdm progress bar."""
        tqdm_bars[0].close()

    def _callback(cond: bool, func: Callable, arg: Any) -> None:
        """Callback a function for a given argument if a condition is true."""
        dummy_result = 0

        def _do_callback(_) -> int:
            """Perform the callback."""
            return host_callback.id_tap(func, arg, result=dummy_result)

        def _not_callback(_) -> int:
            """Do nothing."""
            return dummy_result

        _ = lax.cond(cond, _do_callback, _not_callback, operand=None)

    def _update_progress_bar(loss_val: Float[Array, "1"], iter_num: int) -> None:
        """Updates tqdm progress bar of a JAX scan or loop."""

        # Conditions for iteration number
        is_first: bool = iter_num == 0
        is_multiple: bool = (iter_num % log_rate == 0) & (
            iter_num != n_iters - remainder
        )
        is_remainder: bool = iter_num == n_iters - remainder
        is_last: bool = iter_num == n_iters - 1

        # Define progress bar, if first iteration
        _callback(is_first, _define_tqdm, None)

        # Update progress bar, if multiple of log_rate
        _callback(is_multiple, _update_tqdm, (loss_val, log_rate))

        # Update progress bar, if remainder
        _callback(is_remainder, _update_tqdm, (loss_val, remainder))

        # Close progress bar, if last iteration
        _callback(is_last, _close_tqdm, None)

    def _progress_bar_scan(body_fun: Callable) -> Callable:
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`."""

        def wrapper_progress_bar(carry: Any, x: Union[tuple, int]) -> Any:

            # Get iteration number
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x

            # Compute iteration step
            result = body_fun(carry, x)

            # Get loss value
            *_, loss_val = result

            # Update progress bar
            _update_progress_bar(loss_val, iter_num)

            return result

        return wrapper_progress_bar

    return _progress_bar_scan


__all__ = [
    "fit",
    "fit_natgrads",
    "get_batch",
    "natural_gradients",
    "progress_bar_scan",
]
