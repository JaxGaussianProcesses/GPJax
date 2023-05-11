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
import dataclasses
from functools import partial

from beartype.typing import (
    Callable,
    List,
    Union,
)
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)

from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from gpjax.typing import Array


@dataclasses.dataclass
class AbstractMeanFunction(Module):
    r"""Mean function that is used to parameterise the Gaussian process."""

    @abc.abstractmethod
    def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N 1"]:
        r"""Evaluate the mean function at the given points. This method is required for all subclasses.

        Args:
            x (Float[Array, " D"]): The point at which to evaluate the mean function.

        Returns
        -------
            Float[Array, "1]: The evaluated mean function.
        """
        raise NotImplementedError

    def __add__(
        self, other: Union["AbstractMeanFunction", Float[Array, "1"]]
    ) -> "AbstractMeanFunction":
        r"""Add two mean functions.

        Args:
            other (AbstractMeanFunction): The other mean function to add.

        Returns
        -------
            AbstractMeanFunction: The sum of the two mean functions.
        """
        if isinstance(other, AbstractMeanFunction):
            return SumMeanFunction([self, other])

        return SumMeanFunction([self, Constant(other)])

    def __radd__(
        self,
        other: Union[
            "AbstractMeanFunction", Float[Array, "1"]
        ],  # TODO should this be ScalarFloat? or Num?
    ) -> "AbstractMeanFunction":
        r"""Add two mean functions.

        Args:
            other (AbstractMeanFunction): The other mean function to add.

        Returns
        -------
            AbstractMeanFunction: The sum of the two mean functions.
        """
        return self.__add__(other)

    def __mul__(
        self,
        other: Union[
            "AbstractMeanFunction", Float[Array, "1"]
        ],  # TODO should this be ScalarFloat? or Num?
    ) -> "AbstractMeanFunction":
        r"""Multiply two mean functions.

        Args:
            other (AbstractMeanFunction): The other mean function to multiply.

        Returns
        -------
            AbstractMeanFunction: The product of the two mean functions.
        """
        if isinstance(other, AbstractMeanFunction):
            return ProductMeanFunction([self, other])

        return ProductMeanFunction([self, Constant(other)])

    def __rmul__(
        self,
        other: Union[
            "AbstractMeanFunction", Float[Array, "1"]
        ],  # TODO should this be ScalarFloat? or Num?
    ) -> "AbstractMeanFunction":
        r"""Multiply two mean functions.

        Args:
            other (AbstractMeanFunction): The other mean function to multiply.

        Returns
        -------
            AbstractMeanFunction: The product of the two mean functions.
        """
        return self.__mul__(other)


@dataclasses.dataclass
class Constant(AbstractMeanFunction):
    r"""Constant mean function.

    A constant mean function. This function returns a repeated scalar value for all
    inputs.  The scalar value itself can be treated as a model hyperparameter and
    learned during training but defaults to 1.0.
    """

    constant: Float[Array, "1"] = param_field(jnp.array([0.0]))

    def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N 1"]:
        r"""Evaluate the mean function at the given points.

        Args:
            x (Float[Array, " D"]): The point at which to evaluate the mean function.

        Returns
        -------
            Float[Array, "1"]: The evaluated mean function.
        """
        return jnp.ones((x.shape[0], 1)) * self.constant


@dataclasses.dataclass
class CombinationMeanFunction(AbstractMeanFunction):
    r"""A base class for products or sums of AbstractMeanFunctions."""

    means: List[AbstractMeanFunction]
    operator: Callable = static_field()

    def __init__(
        self,
        means: List[AbstractMeanFunction],
        operator: Callable,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Add means to a list, flattening out instances of this class therein, as in GPFlow kernels.
        items_list: List[AbstractMeanFunction] = []

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

    def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N 1"]:
        r"""Evaluate combination kernel on a pair of inputs.

        Args:
            x (Float[Array, " D"]): The point at which to evaluate the mean function.

        Returns
        -------
            Float[Array, " Q"]: The evaluated mean function.
        """
        return self.operator(jnp.stack([m(x) for m in self.means]))


SumMeanFunction = partial(CombinationMeanFunction, operator=partial(jnp.sum, axis=0))
ProductMeanFunction = partial(
    CombinationMeanFunction, operator=partial(jnp.sum, axis=0)
)
Zero = partial(Constant, constant=jnp.array([0.0]))
