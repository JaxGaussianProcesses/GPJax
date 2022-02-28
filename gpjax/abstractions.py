import typing as tp

import jax
import jax.numpy as jnp
import optax
from optax._src.base import GradientTransformation
from tqdm import trange


def fit(
    objective: tp.Callable,
    params: dict,
    opt_init: tp.Callable,
    opt_update: tp.Callable,
    get_params: tp.Callable,
    n_iters: int = 100,
    log_rate: int = 10,
) -> tp.Dict:
    """Abstracted method for fitting a GP model with respect to a supplied objective function.
    Optimisers used here should originate from Jax's experimental module.

    Args:
        objective (tp.Callable): The objective function that we are optimising with respect to.
        params (dict): The parameters for which we would like to minimise our objective function wtih.
        opt_init (tp.Callable): The supplied optimiser's initialisation function.
        opt_update (tp.Callable): Optimiser's update method.
        get_params (tp.Callable): Return the current parameter state set from the optimiser.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.

    Returns:
        tp.Dict: An optimised set of parameters.
    """
    opt_state = opt_init(params)

    def step(i, opt_state):
        params = get_params(opt_state)
        v, g = jax.value_and_grad(objective)(params)
        return opt_update(i, g, opt_state), v

    tr = trange(n_iters)
    for i in tr:
        opt_state, val = step(i, opt_state)
        if i % log_rate == 0 or i == n_iters:
            tr.set_postfix({"Objective": jnp.round(val, 2)})
    return get_params(opt_state)


def optax_fit(
    objective: tp.Callable,
    params: dict,
    optax_optim: GradientTransformation,
    n_iters: int = 100,
    log_rate: int = 10,
) -> tp.Dict:
    """Abstracted method for fitting a GP model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.

    Args:
        objective (tp.Callable): The objective function that we are optimising with respect to.
        params (dict): The parameters for which we would like to minimise our objective function wtih.
        optax_optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.

    Returns:
        tp.Dict: An optimised set of parameters.
    """
    opt_state = optax_optim.init(params)
    ps = params

    def step(params, opt_state):
        v, g = jax.value_and_grad(objective)(params)
        updates, opt_state = optax_optim.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, v

    tr = trange(n_iters)
    for i in tr:
        params, opt_state, val = step(params, opt_state)
        if i % log_rate == 0 or i == n_iters:
            tr.set_postfix({"Objective": jnp.round(val, 2)})
    return params
