import typing as tp

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from chex import dataclass
from jax import lax
from jax.experimental import host_callback
from jaxtyping import f64
from tqdm.auto import tqdm

from .parameters import trainable_params
from .types import Dataset, PRNGKeyType


@dataclass(frozen=True)
class InferenceState:
    params: tp.Dict
    history: f64["n_iters"]

    def unpack(self):
        return self.params, self.history


def progress_bar_scan(n_iters: int, log_rate: int):
    """Progress bar for Jax.lax scans (adapted from https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/)."""

    tqdm_bars = {}
    remainder = n_iters % log_rate

    def _define_tqdm(args, transform):
        tqdm_bars[0] = tqdm(range(n_iters))

    def _update_tqdm(args, transform):
        loss_val, arg = args
        tqdm_bars[0].update(arg)
        tqdm_bars[0].set_postfix({"Objective": f"{loss_val: .2f}"})

    def _update_progress_bar(loss_val, i):
        """Updates tqdm progress bar of a JAX scan or loop."""
        _ = lax.cond(
            i == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=i),
            lambda _: i,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (i % log_rate == 0) & (i != n_iters - remainder),
            lambda _: host_callback.id_tap(
                _update_tqdm, (loss_val, log_rate), result=i
            ),
            lambda _: i,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm by `remainder`
            i == n_iters - remainder,
            lambda _: host_callback.id_tap(
                _update_tqdm, (loss_val, remainder), result=i
            ),
            lambda _: i,
            operand=None,
        )

    def _close_tqdm(args, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, i):
        return lax.cond(
            i == n_iters - 1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`."""

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            result = func(carry, x)
            *_, loss_val = result
            _update_progress_bar(loss_val, iter_num)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan


def fit(
    objective: tp.Callable,
    params: tp.Dict,
    trainables: tp.Dict,
    optax_optim,
    n_iters: int = 100,
    log_rate: int = 10,
) -> InferenceState:
    """Abstracted method for fitting a GP model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.
    Args:
        objective (tp.Callable): The objective function that we are optimising with respect to.
        params (dict): The parameters for which we would like to minimise our objective function with.
        trainables (dict): Boolean dictionary of same structure as 'params' that determines which parameters should be trained.
        optax_optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.
    Returns:
        tp.Tuple[tp.Dict, f64["n_iters"]]: A tuple comprising optimised parameters and training history respectively.
    """
    opt_state = optax_optim.init(params)

    def loss(params):
        params = trainable_params(params, trainables)
        return objective(params)

    iter_nums = jnp.arange(n_iters)

    @progress_bar_scan(n_iters, log_rate)
    def step(carry, iter_num):
        params, opt_state = carry
        loss_val, loss_gradient = jax.value_and_grad(loss)(params)
        updates, opt_state = optax_optim.update(loss_gradient, opt_state, params)
        params = optax.apply_updates(params, updates)
        carry = params, opt_state
        return carry, loss_val

    (params, _), history = jax.lax.scan(step, (params, opt_state), iter_nums)
    inf_state = InferenceState(params=params, history=history)
    return inf_state


def fit_batches(
    objective: tp.Callable,
    params: tp.Dict,
    trainables: tp.Dict,
    train_data: Dataset,
    optax_optim,
    key: PRNGKeyType,
    batch_size: int,
    n_iters: tp.Optional[int] = 100,
    log_rate: tp.Optional[int] = 10,
) -> InferenceState:
    """Abstracted method for fitting a GP model with mini-batches respect to a supplied objective function.
    Optimisers used here should originate from Optax.
    Args:
        objective (tp.Callable): The objective function that we are optimising with respect to.
        params (dict): The parameters for which we would like to minimise our objective function with.
        trainables (dict): Boolean dictionary of same structure as 'params' that determines which parameters should be trained.
        train_data (Dataset): The training dataset.
        optax_optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        key (PRNGKeyType): The PRNG key for the mini-batch sampling.
        batch_size(int): The batch_size.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.
    Returns:
        tp.Tuple[tp.Dict, f64["n_iters"]]: A tuple comprising optimised parameters and training history respectively.
    """

    opt_state = optax_optim.init(params)

    def loss(params, batch):
        params = trainable_params(params, trainables)
        return objective(params, batch)

    keys = jax.random.split(key, n_iters)
    iter_nums = jnp.arange(n_iters)

    @progress_bar_scan(n_iters, log_rate)
    def step(carry, iter_num__and__key):
        iter_num, key = iter_num__and__key
        params, opt_state = carry

        batch = get_batch(train_data, batch_size, key)

        loss_val, loss_gradient = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optax_optim.update(loss_gradient, opt_state, params)
        params = optax.apply_updates(params, updates)

        carry = params, opt_state
        return carry, loss_val

    (params, _), history = jax.lax.scan(step, (params, opt_state), (iter_nums, keys))
    inf_state = InferenceState(params=params, history=history)
    return inf_state


def get_batch(train_data: Dataset, batch_size: int, key: PRNGKeyType) -> Dataset:
    """Batch the data into mini-batches.
    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.
    Returns:
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    indicies = jr.choice(key, n, (batch_size,), replace=True)

    return Dataset(X=x[indicies], y=y[indicies])
