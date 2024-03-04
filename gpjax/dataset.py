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
import matplotlib.pyplot as plt

from gpjax.typing import Array


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
    X3d_raw: Num[Array, "N D L"] = None
    X2d_raw: Num[Array, "N D"] = None
    Xstatic_raw: Num[Array, "N D"] = None
    y_raw: Num[Array, "N 1"] = None
    names_2d: List[str] = None
    names_3d:  List[str] = None
    names_static:  List[str] = None
    mean_standardization: bool = False
    

    def __post_init__(self) -> None:
        _check_precision(self.X2d_raw, self.y_raw)
        _check_precision(self.Xstatic_raw, self.y_raw)
        _check_precision(self.X3d_raw, self.y_raw)
        
        
        if not self.mean_standardization:
            print("standardized inputs with max and min")
            X3d_min = jnp.min(self.X3d_raw, axis=(0))
            X3d_max = jnp.max(self.X3d_raw,axis=(0))
            X3d = (self.X3d_raw-X3d_min[None,:,:]) / (X3d_max[None,:,:] - X3d_min[None,:,:])
            #X3d = X3d - jnp.mean(X3d, 0)
            X2d_min = jnp.min(self.X2d_raw, axis=0)
            X2d_max = jnp.max(self.X2d_raw,axis=0)
            X2d = (self.X2d_raw - X2d_min) / (X2d_max - X2d_min)
            Xstatic_min = jnp.min(self.Xstatic_raw, axis=0)
            Xstatic_max = jnp.max(self.Xstatic_raw,axis=0)
            Xstatic = (Xstatic_max - self.Xstatic_raw) / (Xstatic_max - Xstatic_min)
        else:
            print("standardized inputs to be Gaussian")
            X3d_mean = jnp.means(self.X3d_raw,axis=(0))
            # X3d_std = jnp.std(X3d, axis=(0,2))
            # X3d = (X3d - X3d_mean[None,:,:]) / X3d_std[None,:,None]
            X3d_std = jnp.std(self.X3d_raw, axis=(0))
            X3d = (self.X3d_raw - X3d_mean[None,:,:]) / X3d_std[None,:,:]
            X2d_std = jnp.std(self.X2d_raw, axis=0)
            X2d_mean = jnp.mean(self.X2d_raw,axis=0)
            X2d = (self.X2d_raw - X2d_mean) / X2d_std
            Xstatic_std = jnp.std(self.Xstatic_raw, axis=0)
            Xstatic_mean = jnp.mean(self.Xstatic_raw,axis=0)
            Xstatic = (self.Xstatic_raw - Xstatic_mean) / Xstatic_std


        print(f"then standardized Y as Gaussian")
        self.Y_mean = jnp.mean(self.y_raw)
        self.Y_std = jnp.std(self.y_raw)
        Y = (self.y_raw - self.Y_mean) / self.Y_std
        plt.hist(Y.T)
        plt.title("Y")

        for data in [self.X3d_raw, X3d]:
            fig, ax = plt.subplots(nrows=3, ncols=4)
            i,j=0,0
            for row in ax:
                for col in row:
                    col.boxplot(data[:,i,:].T, showfliers=False);
                    col.set_title(self.names_3d[i])
                    i+=1
                    if i==data.shape[1]:
                        break
                if i==data.shape[1]:
                    break

        fig, ax = plt.subplots(nrows=1, ncols=X2d.shape[1])
        i=0
        for col in ax:
            col.hist(X2d[:100000,i].T);
            col.set_title(self.names_2d[i])
            i+=1
        fig, ax = plt.subplots(nrows=1, ncols=Xstatic.shape[1])
        i=0
        for col in ax:
            col.hist(Xstatic[:100000,i].T);
            col.set_title(self.names_static[i])
            i+=1

        self.X3d = X3d
        self.X2d = X2d
        self.Xstatic = Xstatic
        self.y = Y
        
        
        


    @property
    def X(self):
        return NotImplementedError("Use X2d, X3d or Xstatic instead")

    @property
    def n(self):
        return self.X2d.shape[0]

    @property
    def dim(self):
        return self.X2d.shape[1] + self.X3d.shape[1] + self.Xstatic.shape[1]

__all__ = [
    "Dataset",
]


