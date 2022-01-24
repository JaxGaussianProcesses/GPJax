import typing as tp

import jax
import jax.numpy as jnp
import optax
from tqdm import trange


def fit(
    objective,
    params: dict,
    opt_init,
    opt_update,
    get_params,
    n_iters: int = 100,
    log_rate: int = 10,
) -> tp.Dict:
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
    objective,
    params: dict,
    optax_optim,
    n_iters: int = 100,
    log_rate: int = 10,
) -> tp.Dict:
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
