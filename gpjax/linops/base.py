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
from dataclasses import dataclass
from numbers import Number

from beartype.typing import (
    Any,
    Generic,
    Iterable,
    Mapping,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import jax.numpy as jnp
from jaxtyping import Float
from plum import (
    Dispatcher,
    ModuleType,
)
from simple_pytree import (
    Pytree,
    static_field,
)
from simple_pytree.pytree import PytreeMeta

from gpjax.typing import (
    Array,
    ScalarFloat,
)

# Generic type.
T = TypeVar("T")

# Generic nested type.
NestedT = Union[T, Iterable["NestedT"], Mapping[Any, "NestedT"]]

# Nested types.
DTypes = Union[Type[jnp.float32], Type[jnp.float64], Type[jnp.int32], Type[jnp.int64]]
ShapeT = TypeVar("ShapeT", bound=NestedT[Tuple[int, ...]])
DTypeT = TypeVar("DTypeT", bound=NestedT[DTypes])

# The dispatcher is used to dispatch the arithmetic methods of the LinearOperator.
_dispatch = Dispatcher()

# The dispatched methods.
_DISPATCHED_METHODS = ["__add__", "__matmul__", "__mul__"]


# The _DispatchDict is used to dispatch the arithmetic methods of the LinearOperator.
class _DispatchDict(dict):
    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item in the dictionary.

        If the key is in the list of dispatched methods, then the corresponding method is dispatched.

        Args:
            key (str): The key of the item.
            value (Any): The value of the item.
        """

        # TODO: Shape checking on arithmetic methods.

        if key in _DISPATCHED_METHODS:
            value = _dispatch(value)

        super().__setitem__(key, value)


class _DispatchMeta(PytreeMeta):
    """Metaclass for dispatching the methods of the AbstractLinearOperator."""

    @classmethod
    def __prepare__(
        cls, __name: str, __bases: Tuple[Type, ...], **kwargs: Any
    ) -> _DispatchDict:
        """Prepare the namespace for the class.

        We use a custom dict for the namespace that dispatches the arithmetic methods.

        Args:
            __name (str): The name of the class.
            __bases (Tuple[Type, ...]): The base classes of the class.
            **kwargs: Keyword arguments.

        Returns:
            _DispatchDict: The namespace for the class.
        """
        return _DispatchDict()

    def __new__(
        cls,
        __name: str,
        __bases: Tuple[Type, ...],
        __namespace: _DispatchDict,
        **kwargs,
    ) -> Type:
        """Create the class.

        Args:
            __name (str): The name of the class.
            __bases (Tuple[Type, ...]): The base classes of the class.
            __namespace (_DispatchDict): The namespace for the class.
            **kwargs: Keyword arguments.

        Returns:
            Type: The class.
        """
        # The "_DispatchDict" is unpacked here, to avoid re-dispatching the arithmetic methods.
        return super().__new__(cls, __name, __bases, {**__namespace}, **kwargs)


# These ModuleTypes are used to dispatch the methods of the LinearOperator.
# In particular, they are used to avoid circular imports.
Zero = ModuleType("gpjax.linops.zero", "Zero")

Identity = ModuleType("gpjax.linops.identity", "Identity")

Diagonal = ModuleType("gpjax.linops.diagonal", "Diagonal")

Dense = ModuleType("gpjax.linops.dense", "Dense")

LowerTriangular = ModuleType("gpjax.linops.lower_triangular", "LowerTriangular")


# The Generic type is used for type checking the LinearOperator's shape and datatype.
# `static_field` is used to mark nodes of the PyTree that don't change under JAX transformations.
# this is important, so that we e.g., don't take the gradient with respect to the shape!
@dataclass
class AbstractLinearOperator(Pytree, Generic[ShapeT, DTypeT], metaclass=_DispatchMeta):
    """Abstract class for linear operators."""

    shape: ShapeT = static_field()
    dtype: DTypeT = static_field()

    @property
    def ndim(self) -> int:
        """Linear operator dimension."""
        return len(self.shape)

    @property
    def T(self) -> "AbstractLinearOperator":
        """Transpose linear operator."""
        from gpjax.linops.dense import Dense

        return Dense(self.to_dense().T)

    def __add__(  # noqa: F821
        self, other: Union[Float[Array, "*"], Number]
    ) -> Float[Array, "*"]:
        from gpjax.linops.utils import (
            to_dense,
            to_linear_operator,
        )

        return to_dense(self + to_linear_operator(other))

    def add(self, other) -> "AbstractLinearOperator":
        """Add a linear operator and another object. A linear operator is returned.

        Args:
            other: The other object.

        Returns:
            AbstractLinearOperator: The sum of the linear operator and the other object.
        """
        from gpjax.linops.utils import to_linear_operator

        return self + to_linear_operator(other)

    def sub(self, other) -> "AbstractLinearOperator":
        """Subtract a linear operator and another object. A linear operator is returned.

        Args:
            other: The other object.

        Returns:
            AbstractLinearOperator: The difference of the linear operator and the other object.
        """
        from gpjax.linops.utils import to_linear_operator

        return self - to_linear_operator(other)

    def matmul(self, other) -> "AbstractLinearOperator":
        """Multiply a linear operator and another object. A linear operator is returned.

        Args:
            other: The other object.

        Returns:
            AbstractLinearOperator: The product of the linear operator and the other object.
        """
        from gpjax.linops.utils import to_linear_operator

        return self.__matmul__(to_linear_operator(other))

    def mul(self, other) -> "AbstractLinearOperator":
        """Multiply a linear operator and another object. A linear operator is returned.

        Args:
            other: The other object.

        Returns:
            AbstractLinearOperator: The product of the linear operator and the other object.
        """
        from gpjax.linops.utils import to_linear_operator

        return self * to_linear_operator(other)

    def __matmul__(self, other: Float[Array, "*"]) -> Float[Array, "*"]:  # noqa: F821
        from gpjax.linops.utils import (
            to_dense,
            to_linear_operator,
        )

        return to_dense(self @ to_linear_operator(other))

    def __mul__(  # noqa: F821
        self, other: Union[Float[Array, "*"], Number]
    ) -> Float[Array, "*"]:
        from gpjax.linops.utils import (
            to_dense,
            to_linear_operator,
        )

        return to_dense(self * to_linear_operator(other))

    def __add__(
        self, other: "AbstractLinearOperator"
    ) -> "AbstractLinearOperator":  # noqa: F821
        from gpjax.linops.dense import Dense

        return Dense(matrix=self.to_dense() + other.to_dense())

    def __add__(self, other: Zero) -> "AbstractLinearOperator":  # noqa: F821
        return self

    def __matmul__(self, other: Diagonal) -> "AbstractLinearOperator":
        from gpjax.linops.dense import Dense

        return Dense(self.to_dense() * other.diagonal())

    def __add__(self, other: Diagonal) -> "AbstractLinearOperator":
        from gpjax.linops.dense import Dense

        diag_indices = jnp.diag_indices(self.shape[0])
        self_plus_other = self.to_dense().at[diag_indices].add(other.diagonal())
        return Dense(matrix=self_plus_other)

    def __matmul__(
        self, other: "AbstractLinearOperator"
    ) -> "AbstractLinearOperator":  # noqa: F821
        from gpjax.linops.dense import Dense

        return Dense(matrix=(self.to_dense() @ other.to_dense()))

    def __matmul__(self, other: Identity) -> "AbstractLinearOperator":  # noqa: F821
        return self

    def __matmul__(self, other: Zero) -> "AbstractLinearOperator":  # noqa: F821
        from gpjax.linops.zero import Zero

        return Zero((self.shape[0], other.shape[1]), dtype=self.dtype)

    def __mul__(
        self, other: "AbstractLinearOperator"
    ) -> "AbstractLinearOperator":  # noqa: F821
        """Multiply linear operator by scalar."""
        from gpjax.linops.dense import Dense

        return Dense(matrix=self.to_dense() * other.to_dense())

    def __mul__(self, other: Zero) -> "AbstractLinearOperator":  # noqa: F821
        from gpjax.linops.zero import Zero

        return Zero(shape=self.shape, dtype=self.dtype)

    def __sub__(
        self, other: Union["AbstractLinearOperator", Float[Array, "N M"], Number]
    ) -> "AbstractLinearOperator":
        """Subtract linear operator."""
        return self + (other * -1)

    def __rsub__(
        self, other: Union["AbstractLinearOperator", Float[Array, "N M"], Number]
    ) -> "AbstractLinearOperator":
        """Reimplimentation of subtract linear operator."""
        return (self * -1) + other

    def __radd__(
        self, other: Union["AbstractLinearOperator", Float[Array, "N M"], Number]
    ) -> "AbstractLinearOperator":
        """Reimplimentation of add linear operator."""
        return self + other

    def __rmul__(
        self, other: Union["AbstractLinearOperator", Float[Array, "N M"], Number]
    ) -> "AbstractLinearOperator":
        """Reimplimentation of multiply linear operator by scalar."""
        return self.__mul__(other)

    def __rmatmul__(
        self, other: Union["AbstractLinearOperator", Float[Array, "N M"]]
    ) -> Union["AbstractLinearOperator", Float[Array, "N M"]]:
        """Reimplimentation of matrix multiplication."""
        if other.ndim == 1:
            return self.T @ other
        return (self.T @ other.T).T

    def diagonal(self) -> Float[Array, " N"]:
        """
        Diagonal of the covariance operator.

        Returns
        -------
            Float[Array, " N"]: The diagonal of the covariance operator.
        """
        return jnp.diag(self.to_dense())

    def trace(self) -> ScalarFloat:
        """Trace of the linear matrix.

        Returns
        -------
            ScalarFloat: Trace of the linear matrix.
        """
        return jnp.sum(self.diagonal())

    def log_det(self) -> ScalarFloat:
        """Log determinant of the linear matrix. Default implementation uses dense Cholesky decomposition.

        Returns
        -------
            ScalarFloat: Log determinant of the linear matrix.
        """
        root = self.to_root()
        return 2.0 * jnp.sum(jnp.log(root.diagonal()))

    def to_root(self) -> LowerTriangular:
        """Compute the root of the linear operator via the Cholesky decomposition.

        Returns
        -------
            Float[Array, "N N"]: Lower Cholesky decomposition of the linear operator.
        """
        from gpjax.linops.triangular import LowerTriangular

        L = jnp.linalg.cholesky(self.to_dense())

        return LowerTriangular.from_dense(L)

    def inverse(self) -> Dense:
        """Inverse of the linear matrix. Default implementation uses dense Cholesky decomposition.

        Returns
        -------
            AbstractLinearOperator: Inverse of the linear matrix.
        """
        from gpjax.linops.dense import Dense

        n = self.shape[0]
        return Dense(self.solve(jnp.eye(n)))

    def solve(self, rhs: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Solve linear system. Default implementation uses dense Cholesky decomposition.

        Args:
            rhs (Float[Array, "N M"]): Right hand side of the linear system.

        Returns
        -------
            Float[Array, "N M"]: Solution of the linear system.
        """
        root = self.to_root()
        rootT = root.T

        return rootT.solve(root.solve(rhs))

    @abc.abstractmethod
    def to_dense(self) -> Float[Array, "..."]:
        """Construct dense matrix from the linear operator.

        Returns
        -------
            Float[Array, "..."]: Dense linear matrix.
        """
        raise NotImplementedError

    @classmethod
    def from_dense(cls, dense: Float[Array, "N M"]) -> "AbstractLinearOperator":
        """Construct linear operator from dense matrix.

        Args:
            dense (Float[Array, "N N"]): Dense matrix.

        Returns
        -------
            AbstractLinearOperator: Linear operator.
        """
        raise NotImplementedError


__all__ = [
    "AbstractLinearOperator",
]
