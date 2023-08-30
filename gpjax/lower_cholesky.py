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

import cola
import jax.numpy as jnp

# TODO: Once this functionality is supported in CoLA, remove this.


@cola.dispatch
def lower_cholesky(A: cola.ops.LinearOperator):  # noqa: F811
    """Returns the lower Cholesky factor of a linear operator.

    Args:
        A (cola.ops.LinearOperator): A linear operator.

    Returns:
        cola.ops.LinearOperator: The lower Cholesky factor of A.
    """

    if cola.PSD not in A.annotations:
        raise ValueError(
            "Expected LinearOperator to be PSD, did you forget to use cola.PSD?"
        )

    return cola.ops.Triangular(jnp.linalg.cholesky(A.to_dense()), lower=True)


@lower_cholesky.dispatch
def _(A: cola.ops.Diagonal):  # noqa: F811
    return cola.ops.Diagonal(jnp.sqrt(A.diag))


@lower_cholesky.dispatch
def _(A: cola.ops.Identity):  # noqa: F811
    return A


@lower_cholesky.dispatch
def _(A: cola.ops.Kronecker):  # noqa: F811
    return cola.ops.Kronecker(*[lower_cholesky(Ai) for Ai in A.Ms])


@lower_cholesky.dispatch
def _(A: cola.ops.BlockDiag):  # noqa: F811
    return cola.ops.BlockDiag(
        *[lower_cholesky(Ai) for Ai in A.Ms], multiplicities=A.multiplicities
    )
