import typing as tp

import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm
from jax import lax
from jax.experimental import host_callback

from .parameters import trainable_params
from .types import Dataset


def progress_bar_scan(n_iters, log_rate):
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
            i = x
            result = func(carry, x)
            *_, loss_val = result
            _update_progress_bar(loss_val, i)
            return close_tqdm(result, i)

        return wrapper_progress_bar

    return _progress_bar_scan


def fit(
    objective: tp.Callable,
    params: tp.Dict,
    trainables: tp.Dict,
    optax_optim,
    n_iters: int = 100,
    log_rate: int = 10,
) -> tp.Dict:
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
        tp.Dict: An optimised set of parameters.
    """
    opt_state = optax_optim.init(params)

    def loss(params):
        params = trainable_params(params, trainables)
        return objective(params)

    @progress_bar_scan(n_iters, log_rate)
    def step(params_opt_state, i):
        params, opt_state = params_opt_state
        loss_val, loss_gradient = jax.value_and_grad(loss)(params)
        updates, opt_state = optax_optim.update(loss_gradient, opt_state, params)
        params = optax.apply_updates(params, updates)
        params_opt_state = params, opt_state
        return params_opt_state, loss_val

    (params, _), _ = jax.lax.scan(step, (params, opt_state), jnp.arange(n_iters))
    return params


def fit_batches(
    objective: tp.Callable,
    params: tp.Dict,
    trainables: tp.Dict,
    train_data: Dataset,
    optax_optim,
    n_iters: tp.Optional[int] = 100,
    log_rate: tp.Optional[int] = 10,
) -> tp.Dict:
    """Abstracted method for fitting a GP model with mini-batches respect to a supplied objective function.
    Optimisers used here should originate from Optax.
    Args:
        objective (tp.Callable): The objective function that we are optimising with respect to.
        params (dict): The parameters for which we would like to minimise our objective function with.
        trainables (dict): Boolean dictionary of same structure as 'params' that determines which parameters should be trained.
        train_data (Dataset): The training dataset.
        opt_init (tp.Callable): The supplied optimiser's initialisation function.
        opt_update (tp.Callable): Optimiser's update method.
        get_params (tp.Callable): Return the current parameter state set from the optimiser.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.
    Returns:
        tp.Dict: An optimised set of parameters.
    """
    
    opt_state = optax_optim.init(params)
    next_batch = train_data.get_batcher()

    def loss(params, batch):
        params = trainable_params(params, trainables)
        return objective(params, batch)

    @progress_bar_scan(n_iters, log_rate)
    def step(params_opt_state, i):
        params, opt_state = params_opt_state
        batch = next_batch()
        loss_val, loss_gradient = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optax_optim.update(loss_gradient, opt_state, params)
        params = optax.apply_updates(params, updates)
        params_opt_state = params, opt_state
        return params_opt_state, loss_val

    (params, _), _ = jax.lax.scan(step, (params, opt_state), jnp.arange(n_iters))

    return params
