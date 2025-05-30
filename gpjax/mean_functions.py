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


import abc
import functools as ft

import beartype.typing as tp
from flax import nnx
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)

from gpjax.parameters import (
    Parameter,
    Real,
    Static,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class AbstractMeanFunction(nnx.Module):
    r"""Mean function that is used to parameterise the Gaussian process."""

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N O"]:
        r"""Evaluate the mean function at the given points. This method is required for all subclasses.

        Args:
            x (Float[Array, " D"]): The point at which to evaluate the mean function.

        Returns:
            Float[Array, "1]: The evaluated mean function.
        """
        raise NotImplementedError

    def __add__(
        self, other: tp.Union["AbstractMeanFunction", Float[Array, " O"]]
    ) -> "AbstractMeanFunction":
        r"""Add two mean functions.

        Args:
            other (AbstractMeanFunction): The other mean function to add.

        Returns:
            AbstractMeanFunction: The sum of the two mean functions.
        """
        if isinstance(other, AbstractMeanFunction):
            return SumMeanFunction([self, other])

        return SumMeanFunction([self, Constant(other)])

    def __radd__(
        self,
        other: tp.Union[
            "AbstractMeanFunction", Float[Array, " O"]
        ],  # TODO should this be ScalarFloat? or Num?
    ) -> "AbstractMeanFunction":
        r"""Add two mean functions.

        Args:
            other (AbstractMeanFunction): The other mean function to add.

        Returns:
            AbstractMeanFunction: The sum of the two mean functions.
        """
        return self.__add__(other)

    def __mul__(
        self,
        other: tp.Union[
            "AbstractMeanFunction", Float[Array, " O"]
        ],  # TODO should this be ScalarFloat? or Num?
    ) -> "AbstractMeanFunction":
        r"""Multiply two mean functions.

        Args:
            other (AbstractMeanFunction): The other mean function to multiply.

        Returns:
            AbstractMeanFunction: The product of the two mean functions.
        """
        if isinstance(other, AbstractMeanFunction):
            return ProductMeanFunction([self, other])

        return ProductMeanFunction([self, Constant(other)])

    def __rmul__(
        self,
        other: tp.Union[
            "AbstractMeanFunction", Float[Array, " O"]
        ],  # TODO should this be ScalarFloat? or Num?
    ) -> "AbstractMeanFunction":
        r"""Multiply two mean functions.

        Args:
            other (AbstractMeanFunction): The other mean function to multiply.

        Returns:
            AbstractMeanFunction: The product of the two mean functions.
        """
        return self.__mul__(other)


class Constant(AbstractMeanFunction):
    r"""Constant mean function.

    A constant mean function. This function returns a repeated scalar value for all
    inputs.  The scalar value itself can be treated as a model hyperparameter and
    learned during training but defaults to 1.0.
    """

    def __init__(
        self,
        constant: tp.Union[ScalarFloat, Float[Array, " O"], Parameter, Static] = 0.0,
    ):
        if isinstance(constant, Parameter) or isinstance(constant, Static):
            self.constant = constant
        else:
            self.constant = Real(jnp.array(constant))

    def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N O"]:
        r"""Evaluate the mean function at the given points.

        Args:
            x (Float[Array, " D"]): The point at which to evaluate the mean function.

        Returns:
            Float[Array, "1"]: The evaluated mean function.
        """
        return jnp.ones((x.shape[0], 1)) * self.constant.value


class Zero(Constant):
    r"""Zero mean function.

    The zero mean function. This function returns a zero scalar value for all
    inputs. Unlike the Constant mean function, the constant scalar zero is fixed, and
    cannot be treated as a model hyperparameter and learned during training.
    """

    def __init__(self):
        super().__init__(constant=Static(jnp.array(0.0)))


class CombinationMeanFunction(AbstractMeanFunction):
    r"""A base class for products or sums of AbstractMeanFunctions."""

    def __init__(
        self,
        means: list[AbstractMeanFunction],
        operator: tp.Callable,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Add means to a list, flattening out instances of this class therein, as in GPFlow kernels.
        items_list: list[AbstractMeanFunction] = []

        for item in means:
            if not isinstance(item, AbstractMeanFunction):
                raise TypeError(
                    "can only combine AbstractMeanFunction instances"
                )  # pragma: no cover

            if isinstance(item, self.__class__):
                items_list.extend(item.means)
            else:
                items_list.append(item)

        self.means = items_list
        self.operator = operator

    def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N O"]:
        r"""Evaluate combination kernel on a pair of inputs.

        Args:
            x (Float[Array, " D"]): The point at which to evaluate the mean function.

        Returns:
            Float[Array, " Q"]: The evaluated mean function.
        """
        return self.operator(jnp.stack([m(x) for m in self.means]))


SumMeanFunction = ft.partial(
    CombinationMeanFunction, operator=ft.partial(jnp.sum, axis=0)
)
ProductMeanFunction = ft.partial(
    CombinationMeanFunction, operator=ft.partial(jnp.prod, axis=0)
)
