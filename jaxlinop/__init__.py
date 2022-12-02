# Copyright 2022 The JaxLinOp Contributors. All Rights Reserved.
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

__version__ = "0.0.3"
__authors__ = "Daniel Dodd, Thomas Pinder"
__emails__ = "d.dodd1@lancaster.ac.uk, tompinder@live.co.uk"
__license__ = "Apache 2.0"
__uri__ = "https://github.com/Daniel-Dodd/jax_linear_operator"
__description__ = "A JaxLinOp library."
__contributors__ = (
    "https://github.com/Daniel-Dodd/jax_linear_operator/graphs/contributors"
)

from .linear_operator import LinearOperator
from .dense_linear_operator import DenseLinearOperator
from .diagonal_linear_operator import DiagonalLinearOperator
from .constant_diagonal_linear_operator import ConstantDiagonalLinearOperator
from .identity_linear_operator import IdentityLinearOperator
from .zero_linear_operator import ZeroLinearOperator
from .triangular_linear_operator import (
    LowerTriangularLinearOperator,
    UpperTriangularLinearOperator,
)
from .utils import (
    identity,
    to_dense,
)

__all__ = [
    "LinearOperator",
    "DenseLinearOperator",
    "DiagonalLinearOperator",
    "ConstantDiagonalLinearOperator",
    "IdentityLinearOperator",
    "ZeroLinearOperator",
    "LowerTriangularLinearOperator",
    "UpperTriangularLinearOperator",
    "identity",
    "to_dense",
]
