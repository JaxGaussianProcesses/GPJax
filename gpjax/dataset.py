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
import warnings

from beartype.typing import Optional
from typing import List 
import jax.numpy as jnp
from jaxtyping import Num
from simple_pytree import Pytree
from gpjax.typing import Array
import gpjax as gpx
from jax import vmap


@dataclass
class Dataset(Pytree):
    r"""Base class for datasets.

    Attributes
    ----------
        X (Optional[Num[Array, "N D"]]): input data.
        y (Optional[Num[Array, "N Q"]]): output data.
    """

    X: Optional[Num[Array, "N D"]] = None
    y: Optional[Num[Array, "N Q"]] = None

    def __post_init__(self) -> None:
        r"""Checks that the shapes of $`X`$ and $`y`$ are compatible,
        and provides warnings regarding the precision of $`X`$ and $`y`$."""
        _check_shape(self.X, self.y)
        _check_precision(self.X, self.y)

    def __repr__(self) -> str:
        r"""Returns a string representation of the dataset."""
        repr = f"- Number of observations: {self.n}\n- Input dimension: {self.in_dim}"
        return repr

    def is_supervised(self) -> bool:
        r"""Returns `True` if the dataset is supervised."""
        return self.X is not None and self.y is not None

    def is_unsupervised(self) -> bool:
        r"""Returns `True` if the dataset is unsupervised."""
        return self.X is None and self.y is not None

    def __add__(self, other: "Dataset") -> "Dataset":
        r"""Combine two datasets. Right hand dataset is stacked beneath the left."""
        X = None
        y = None

        if self.X is not None and other.X is not None:
            X = jnp.concatenate((self.X, other.X))

        if self.y is not None and other.y is not None:
            y = jnp.concatenate((self.y, other.y))

        return Dataset(X=X, y=y)

    @property
    def n(self) -> int:
        r"""Number of observations."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        r"""Dimension of the inputs, $`X`$."""
        return self.X.shape[1]


def _check_shape(
    X: Optional[Num[Array, "..."]], y: Optional[Num[Array, "..."]]
) -> None:
    r"""Checks that the shapes of $`X`$ and $`y`$ are compatible."""
    if X is not None and y is not None and X.shape[0] != y.shape[0]:
        raise ValueError(
            "Inputs, X, and outputs, y, must have the same number of rows."
            f" Got X.shape={X.shape} and y.shape={y.shape}."
        )

    if X is not None and X.ndim != 2:
        raise ValueError(
            f"Inputs, X, must be a 2-dimensional array. Got X.ndim={X.ndim}."
        )

    if y is not None and y.ndim != 2:
        raise ValueError(
            f"Outputs, y, must be a 2-dimensional array. Got y.ndim={y.ndim}."
        )


def _check_precision(
    X: Optional[Num[Array, "..."]], y: Optional[Num[Array, "..."]]
) -> None:
    r"""Checks the precision of $`X`$ and $`y`."""
    if X is not None and X.dtype != jnp.float64:
        warnings.warn(
            "X is not of type float64. "
            f"Got X.dtype={X.dtype}. This may lead to numerical instability. ",
            stacklevel=2,
        )

    if y is not None and y.dtype != jnp.float64:
        warnings.warn(
            "y is not of type float64."
            f"Got y.dtype={y.dtype}. This may lead to numerical instability.",
            stacklevel=2,
        )




@dataclass
class VerticalDataset(Pytree):
    X3d: Num[Array, "N D L"] = None
    X2d: Num[Array, "N D"] = None
    Xstatic: Num[Array, "N D"] = None
    y: Num[Array, "N 1"] = None
    standardize: bool = True
    Y_mean: Num[Array, "1 1"] = None
    Y_std: Num[Array, "1 1"] = None
    

    def __post_init__(self) -> None:
        _check_precision(self.X2d, self.y)
        _check_precision(self.Xstatic, self.y)
        _check_precision(self.X3d, self.y)
        self.X3d_raw = self.X3d
        
        
        if self.standardize:
            print("standardized inputs with max and min")
            X3d_max = jnp.max(self.X3d,axis=(0,2))
            X3d_min = jnp.min(self.X3d, axis=(0,2))
            X3d = (self.X3d-X3d_min[None,:,None]) / (X3d_max[None,:,None] - X3d_min[None,:,None])
            #X3d_max = jnp.max(self.X3d,axis=(0))
            #X3d_min = jnp.min(self.X3d, axis=(0))
            #X3d = (self.X3d-X3d_min[None,:,:]) / (X3d_max[None,:,:] - X3d_min[None,:,:])
            #X3d = X3d - jnp.mean(X3d, 0)
            X2d_min = jnp.min(self.X2d, axis=0)
            X2d_max = jnp.max(self.X2d,axis=0)
            X2d = (self.X2d - X2d_min) / (X2d_max - X2d_min)
            Xstatic_min = jnp.min(self.Xstatic, axis=0)
            Xstatic_max = jnp.max(self.Xstatic,axis=0)
            Xstatic = (self.Xstatic - Xstatic_min) / (Xstatic_max - Xstatic_min)


            print(f"then standardized Y as Gaussian")
            self.Y_mean = jnp.mean(self.y,0)
            self.Y_std = jnp.sqrt(jnp.var(self.y,0))
            y = (self.y-self.Y_mean) / self.Y_std
            
            self.X3d = X3d
            self.X2d = X2d
            self.Xstatic = Xstatic
            self.y = y
            

    @property
    def X(self):
        return NotImplementedError("Use X2d, X3d or Xstatic instead")

    @property
    def n(self):
        return self.X2d.shape[0]

    @property
    def dim(self):
        return self.X2d.shape[1] + self.X3d.shape[1] + self.Xstatic.shape[1]



    def get_subset(self, M: int, space_filling=False,use_output=False, no_3d=True):
        if space_filling:
            if use_output:
                if no_3d:
                    X = jnp.hstack([self.X2d, self.Xstatic, self.y])
                else:
                    X = jnp.hstack([jnp.mean(self.X3d,-1), self.X2d, self.Xstatic, self.y])
            else:
                if no_3d:
                    X = jnp.hstack([self.X2d, self.Xstatic])
                else:
                    X = jnp.hstack([jnp.mean(self.X3d,-1), self.X2d, self.Xstatic])
            assert X.shape[0] > M
            d = X.shape[1]
            #kernel = gpx.kernels.SumKernel(kernels=[gpx.kernels.RBF(active_dims=[i]) for i in range(d)])
            kernel = gpx.kernels.RBF(lengthscale=jnp.array(.1, dtype=jnp.float64))
            chosen_indicies = []  # iteratively store chosen points
            N = X.shape[0]
            c = jnp.zeros((M - 1, N), dtype=jnp.float64)  # [M-1,N]
            d_squared = vmap(lambda x: kernel(x,x),0)(X) # [N]

            chosen_indicies.append(jnp.argmax(d_squared))  # get first element
            for m in range(M - 1):  # get remaining elements
                ix = jnp.array(chosen_indicies[-1], dtype=int) # increment Cholesky with newest point
                newest_point = X[ix]
                d_temp = jnp.sqrt(d_squared[ix])  # [1]

                L = kernel.cross_covariance(X, newest_point[None, :])[:, 0]  # [N]
                if m == 0:
                    e = L / d_temp
                    c = e[None,:]  # [1,N]
                else:
                    c_temp = c[:, ix : ix + 1]  # [m,1]
                    e = (L - jnp.matmul(jnp.transpose(c_temp), c[:m])) / d_temp  # [N]
                    c = jnp.concatenate([c, e], axis=0)  # [m+1, N]
                    # e = tf.squeeze(e, 0)
                d_squared -= e**2
                d_squared = jnp.maximum(d_squared, 1e-50)  # numerical stability
                chosen_indicies.append(jnp.argmax(d_squared))  # get next element as point with largest score
            chosen_indicies = jnp.array(chosen_indicies, dtype=int)
            return VerticalDataset(X3d=self.X3d[chosen_indicies], X2d=self.X2d[chosen_indicies], Xstatic=self.Xstatic[chosen_indicies], y=self.y[chosen_indicies], standardize=False, Y_mean=self.Y_mean, Y_std=self.Y_std)
        else:
            return VerticalDataset(X3d=self.X3d[:M], X2d=self.X2d[:M], Xstatic=self.Xstatic[:M], y=self.y[:M], standardize=False, Y_mean=self.Y_mean, Y_std=self.Y_std)


__all__ = [
    "Dataset",
]


