# Copyright 2023 The GPJax Contributors. All Rights Reserved.
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

from cola.annotations import PSD
from cola.fns import dispatch
from cola.ops.operator_base import LinearOperator
from cola.ops.operators import (
    BlockDiag,
    Diagonal,
    Identity,
    Kronecker,
    Triangular,
)
import jax.numpy as jnp

# TODO: Once this functionality is supported in CoLA, remove this.


@dispatch
def lower_cholesky(A: LinearOperator) -> Triangular:  # noqa: F811
    """Returns the lower Cholesky factor of a linear operator.

    Args:
        A: The input linear operator.

    Returns:
        Triangular: The lower Cholesky factor of A.
    """

    if PSD not in A.annotations:
        raise ValueError(
            "Expected LinearOperator to be PSD, did you forget to use cola.PSD?"
        )

    return Triangular(jnp.linalg.cholesky(A.to_dense()), lower=True)


@lower_cholesky.dispatch
def _(A: Diagonal):  # noqa: F811
    return Diagonal(jnp.sqrt(A.diag))


@lower_cholesky.dispatch
def _(A: Identity):  # noqa: F811
    return A


@lower_cholesky.dispatch
def _(A: Kronecker):  # noqa: F811
    return Kronecker(*[lower_cholesky(Ai) for Ai in A.Ms])


@lower_cholesky.dispatch
def _(A: BlockDiag):  # noqa: F811
    return BlockDiag(
        *[lower_cholesky(Ai) for Ai in A.Ms], multiplicities=A.multiplicities
    )
