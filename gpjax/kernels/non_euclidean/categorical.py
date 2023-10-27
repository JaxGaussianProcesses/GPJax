# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
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


from dataclasses import dataclass
from typing import (
    NamedTuple,
    Union,
)

import jax.numpy as jnp
from jaxtyping import (
    Float,
    Int,
)
import tensorflow_probability.substrates.jax as tfp

from gpjax.base import (
    param_field,
    static_field,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.typing import (
    Array,
    ScalarInt,
)

tfb = tfp.bijectors

CatKernelParams = NamedTuple(
    "CatKernelParams",
    [("stddev", Float[Array, "N 1"]), ("cholesky_lower", Float[Array, " N*(N-1)//2"])],
)


@dataclass
class CatKernel(AbstractKernel):
    r"""The categorical kernel is defined for a fixed number of values of categorical input.

    It stores a standard dev for each input value (i.e. the diagonal of the gram), and a lower cholesky factor for correlations.
    It returns the corresponding values from an the gram matrix when called.

    Args:
        stddev (Float[Array, "N"]): The standard deviation parameters, one for each input space value.
        cholesky_lower (Float[Array, "N*(N-1)//2 N"]): The parameters for the Cholesky factor of the gram matrix.
        inspace_vals (list): The values in the input space this CatKernel works for. Stored for order reference, making clear the indices used for each input space value.
        name (str): The name of the kernel.
        input_1hot (bool): If True, the kernel expect to be called with a 1-hot encoding of the input space values. If False, it expects the indices of the input space values.

    Raises:
        ValueError: If the number of diagonal variance parameters does not match the number of input space values.
    """

    stddev: Float[Array, " N"] = param_field(jnp.ones((2,)), bijector=tfb.Softplus())
    cholesky_lower: Float[Array, "N N"] = param_field(
        jnp.eye(2), bijector=tfb.CorrelationCholesky()
    )
    inspace_vals: Union[list, None] = static_field(None)
    name: str = "Categorical Kernel"
    input_1hot: bool = static_field(False)

    def __post_init__(self):
        if self.inspace_vals is not None and len(self.inspace_vals) != len(self.stddev):
            raise ValueError(
                f"The number of stddev parameters ({len(self.stddev)}) has to match the number of input space values ({len(self.inspace_vals)}), unless inspace_vals is None."
            )

    @property
    def explicit_gram(self) -> Float[Array, "N N"]:
        """Access the PSD gram matrix resulting from the parameters.

        Returns:
            Float[Array, "N N"]: The gram matrix.
        """
        L = self.stddev.reshape(-1, 1) * self.cholesky_lower
        return L @ L.T

    def __call__(  # TODO not consistent with general kernel interface
        self,
        x: Union[ScalarInt, Int[Array, " N"]],
        y: Union[ScalarInt, Int[Array, " N"]],
    ):
        r"""Compute the (co)variance between a pair of dictionary indices.

        Args:
            x (Union[ScalarInt, Int[Array, "N"]]): The index of the first dictionary entry, or its one-hot encoding.
            y (Union[ScalarInt, Int[Array, "N"]]): The index of the second dictionary entry, or its one-hot encoding.

        Returns
        -------
            ScalarFloat: The value of $k(v_i, v_j)$.
        """
        try:
            x = x.squeeze()
            y = y.squeeze()
        except AttributeError:
            pass
        if self.input_1hot:
            return self.explicit_gram[jnp.outer(x, y) == 1]
        else:
            return self.explicit_gram[x, y]

    @staticmethod
    def num_cholesky_lower_params(num_inspace_vals: ScalarInt) -> ScalarInt:
        """Compute the number of parameters required to store the lower triangular Cholesky factor of the gram matrix.

        Args:
            num_inspace_vals (ScalarInt): The number of values in the input space.

        Returns:
            ScalarInt: The number of parameters required to store the lower triangle of the Cholesky factor of the gram matrix.
        """
        return num_inspace_vals * (num_inspace_vals - 1) // 2

    @staticmethod
    def gram_to_stddev_cholesky_lower(gram: Float[Array, "N N"]) -> CatKernelParams:
        """Compute the standard deviation and lower triangular Cholesky factor of the gram matrix.

        Args:
            gram (Float[Array, "N N"]): The gram matrix.

        Returns:
            tuple[Float[Array, "N"], Float[Array, "N N"]]: The standard deviation and lower triangular Cholesky factor of the gram matrix, where the latter is scaled to result in unit variances.
        """
        stddev = jnp.sqrt(jnp.diag(gram))
        L = jnp.linalg.cholesky(gram) / stddev.reshape(-1, 1)
        return CatKernelParams(stddev, L)
