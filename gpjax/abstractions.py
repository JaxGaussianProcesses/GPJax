import typing as tp

import jax
import optax
import tensorflow.data as tfd
from tqdm import trange

from .parameters import trainable_params
from .types import Dataset


def fit(
    objective: tp.Callable,
    params: tp.Dict,
    trainables: tp.Dict,
    opt_init,
    opt_update,
    get_params,
    n_iters: int = 100,
    log_rate: int = 10,
    jit_compile: bool = False,
) -> tp.Dict:
    """Abstracted method for fitting a GP model with respect to a supplied objective function.
    Optimisers used here should originate from Jax's experimental module.
    Args:
        objective (tp.Callable): The objective function that we are optimising with respect to.
        params (dict): The parameters for which we would like to minimise our objective function wtih.
        trainables (dict): Boolean dictionary of same structure as 'params' that determines which parameters should be trained.
        opt_init (tp.Callable): The supplied optimiser's initialisation function.
        opt_update (tp.Callable): Optimiser's update method.
        get_params (tp.Callable): Return the current parameter state set from the optimiser.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.
        jit_compile (bool,  optional): Jit compiles the loss function and training step within the training loop.
    Returns:
        tp.Dict: An optimised set of parameters.
    """
    opt_state = opt_init(params)

    def loss(params):
        params = trainable_params(params, trainables)
        return objective(params)

    def step(i, opt_state):
        params = get_params(opt_state)
        loss_val, loss_gradient = jax.value_and_grad(loss)(params)
        return opt_update(i, loss_gradient, opt_state), loss_val

    if jit_compile:
        loss = jax.jit(loss)
        step = jax.jit(step)

    tr = trange(n_iters)
    for i in tr:
        opt_state, val = step(i, opt_state)
        if i % log_rate == 0 or i == n_iters:
            tr.set_postfix({"Objective": f"{val: .2f}"})
    return get_params(opt_state)


def optax_fit(
    objective: tp.Callable,
    params: tp.Dict,
    trainables: tp.Dict,
    optax_optim,
    n_iters: int = 100,
    log_rate: int = 10,
    jit_compile: bool = False,
) -> tp.Dict:
    """Abstracted method for fitting a GP model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.
    Args:
        objective (tp.Callable): The objective function that we are optimising with respect to.
        params (dict): The parameters for which we would like to minimise our objective function wtih.
        trainables (dict): Boolean dictionary of same structure as 'params' that determines which parameters should be trained.
        optax_optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.
        jit_compile (bool,  optional): Jit compiles the loss function and training step within the training loop.
    Returns:
        tp.Dict: An optimised set of parameters.
    """
    opt_state = optax_optim.init(params)

    def loss(params):
        params = trainable_params(params, trainables)
        return objective(params)

    def step(params, opt_state):
        loss_val, loss_gradient = jax.value_and_grad(loss)(params)
        updates, opt_state = optax_optim.update(loss_gradient, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    if jit_compile:
        loss = jax.jit(loss)
        step = jax.jit(step)

    tr = trange(n_iters)
    for i in tr:
        params, opt_state, val = step(params, opt_state)
        if i % log_rate == 0 or i == n_iters:
            tr.set_postfix({"Objective": f"{val: .2f}"})
    return params


# Mini-batch loader from a TensorFlow Dataset:
def batch_loader(data: tfd.Dataset) -> tp.Callable[[None], Dataset]:
    """Abstracted method for loading mini-batches from a TensorFlow Dataset as a GPJax Dataset.
    Args:
        dataset (tfd.Dataset): Training data as a TensorFlow Dataset.
    Returns:
        tp.Callable[[None], Dataset]: A function that produces mini-batches as a GPJax Dataset.
    """

    dataset_iter = iter(data)

    def next_batch() -> Dataset:
        x_batch, y_batch = dataset_iter.next()
        return Dataset(X=x_batch.numpy(), y=y_batch.numpy())

    return next_batch


# Mini-batch gradient descent:
def fit_batches(
    objective: tp.Callable,
    params: tp.Dict,
    trainables: tp.Dict,
    opt_init,
    opt_update,
    get_params,
    get_batch,
    n_iters: tp.Optional[int] = 100,
    log_rate: tp.Optional[int] = 10,
    jit_compile: bool = False,
) -> tp.Dict:
    """Abstracted method for fitting a GP model with mini-batches respect to a supplied objective function.
    Optimisers used here should originate from Jax's experimental module.
    Args:
        objective (tp.Callable): The objective function that we are optimising with respect to.
        params (dict): The parameters for which we would like to minimise our objective function wtih.
        trainables (dict): Boolean dictionary of same structure as 'params' that determines which parameters should be trained.
        opt_init (tp.Callable): The supplied optimiser's initialisation function.
        opt_update (tp.Callable): Optimiser's update method.
        get_params (tp.Callable): Return the current parameter state set from the optimiser.
        n_iters (int, optional): The number of optimisation steps to run. Defaults to 100.
        log_rate (int, optional): How frequently the objective function's value should be printed. Defaults to 10.
        jit_compile (bool,  optional): Jit compiles the loss function and training step within the training loop.
    Returns:
        tp.Dict: An optimised set of parameters.
    """

    opt_state = opt_init(params)

    def loss(params, batch):
        params = trainable_params(params, trainables)
        return objective(params, batch)

    def train_step(i, opt_state, batch):
        params = get_params(opt_state)
        loss_val, loss_gradient = jax.value_and_grad(loss)(params, batch)
        return opt_update(i, loss_gradient, opt_state), loss_val

    if jit_compile:
        loss = jax.jit(loss)
        train_step = jax.jit(train_step)

    tr = trange(n_iters)
    for i in tr:
        batch = get_batch()
        opt_state, val = train_step(i, opt_state, batch)
        if i % log_rate == 0 or i == n_iters:
            tr.set_postfix({"Objective": f"{val: .2f}"})
    return get_params(opt_state)
