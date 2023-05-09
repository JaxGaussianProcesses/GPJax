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

from beartype.typing import (
    Any,
    Callable,
    Union,
)
from jax import lax
from jax.experimental import host_callback
from jaxtyping import Float  # noqa: TCH002
from tqdm.auto import tqdm

from gpjax.typing import Array  # noqa: TCH001


def progress_bar(num_iters: int, log_rate: int) -> Callable:
    r"""Progress bar decorator for the body function of a `jax.lax.scan`.

    Example:
    ```python
        >>> import jax.numpy as jnp
        >>> import jax
        >>>
        >>> carry = jnp.array(0.0)
        >>> iteration_numbers = jnp.arange(100)

        >>> @progress_bar(num_iters=iteration_numbers.shape[0], log_rate=10)
        >>> def body_func(carry, x):
        ...    return carry, x
        >>>
        >>> carry, _ = jax.lax.scan(body_func, carry, iteration_numbers)
    ```

    Adapted from this [excellent blog post](https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/).

    Might be nice in future to directly create a general purpose `verbose scan`
    inplace of a for a jax.lax.scan, that takes the same arguments as a jax.lax.scan,
    but prints a progress bar.
    """
    tqdm_bars = {}
    remainder = num_iters % log_rate

    """Define a tqdm progress bar."""
    tqdm_bars[0] = tqdm(range(num_iters))
    tqdm_bars[0].set_description("Compiling...", refresh=True)

    def _update_tqdm(args: Any, transform: Any) -> None:
        r"""Update the tqdm progress bar with the latest objective value."""
        value, iter_num = args
        tqdm_bars[0].set_description("Running", refresh=False)
        tqdm_bars[0].update(iter_num)
        tqdm_bars[0].set_postfix({"Value": f"{value: .2f}"})

    def _close_tqdm(args: Any, transform: Any) -> None:
        r"""Close the tqdm progress bar."""
        tqdm_bars[0].close()

    def _callback(cond: bool, func: Callable, arg: Any) -> None:
        r"""Callback a function for a given argument if a condition is true."""
        dummy_result = 0

        def _do_callback(_) -> int:
            """Perform the callback."""
            return host_callback.id_tap(func, arg, result=dummy_result)

        def _not_callback(_) -> int:
            """Do nothing."""
            return dummy_result

        _ = lax.cond(cond, _do_callback, _not_callback, operand=None)

    def _update_progress_bar(value: Float[Array, "1"], iter_num: int) -> None:
        r"""Update the tqdm progress bar."""
        # Conditions for iteration number
        is_multiple: bool = (iter_num % log_rate == 0) & (
            iter_num != num_iters - remainder
        )
        is_remainder: bool = iter_num == num_iters - remainder
        is_last: bool = iter_num == num_iters - 1

        # Update progress bar, if multiple of log_rate
        _callback(is_multiple, _update_tqdm, (value, log_rate))

        # Update progress bar, if remainder
        _callback(is_remainder, _update_tqdm, (value, remainder))

        # Close progress bar, if last iteration
        _callback(is_last, _close_tqdm, None)

    def _progress_bar(body_fun: Callable) -> Callable:
        r"""Decorator that adds a progress bar to `body_fun` used in `jax.lax.scan`."""

        def wrapper_progress_bar(carry: Any, x: Union[tuple, int]) -> Any:
            # Get iteration number
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x

            # Compute iteration step
            result = body_fun(carry, x)

            # Get value
            *_, value = result

            # Update progress bar
            _update_progress_bar(value, iter_num)

            return result

        return wrapper_progress_bar

    return _progress_bar


__all__ = [
    "progress_bar",
]
