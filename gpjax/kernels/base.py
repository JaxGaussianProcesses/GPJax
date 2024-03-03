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

import abc
from dataclasses import dataclass
from functools import partial

from beartype.typing import (
    Callable,
    List,
    Optional,
    Union,
)
import jax.numpy as jnp
import numpy as np
import jax
from jaxtyping import (
    Float,
    Num,
)
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

from gpjax.base import (
    Module,
    param_field,
    static_field,
)
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
    ScalarInt,
)


@dataclass
class AbstractKernel(Module):
    r"""Base kernel class."""

    compute_engine: AbstractKernelComputation = static_field(DenseKernelComputation())
    active_dims: Optional[List[int]] = static_field(None)
    name: str = static_field("AbstractKernel")

    @property
    def ndims(self):
        return 1 if not self.active_dims else len(self.active_dims)

    def cross_covariance(self, x: Num[Array, "N D"], y: Num[Array, "M D"]):
        return self.compute_engine.cross_covariance(self, x, y)

    def gram(self, x: Num[Array, "N D"]):
        return self.compute_engine.gram(self, x)

    def slice_input(self, x: Float[Array, "... D"]) -> Float[Array, "... Q"]:
        r"""Slice out the relevant columns of the input matrix.

        Select the relevant columns of the supplied matrix to be used within the
        kernel's evaluation.

        Args:
            x (Float[Array, "... D"]): The matrix or vector that is to be sliced.

        Returns
        -------
            Float[Array, "... Q"]: A sliced form of the input matrix.
        """
        return x[..., self.active_dims] if self.active_dims is not None else x

    @abc.abstractmethod
    def __call__(
        self,
        x: Num[Array, " D"],
        y: Num[Array, " D"],
    ) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Num[Array, " D"]): The left hand input of the kernel function.
            y (Num[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        raise NotImplementedError

    def __add__(self, other: Union["AbstractKernel", ScalarFloat]) -> "AbstractKernel":
        r"""Add two kernels together.
        Args:
            other (AbstractKernel): The kernel to be added to the current kernel.

        Returns
        -------
            AbstractKernel: A new kernel that is the sum of the two kernels.
        """
        if isinstance(other, AbstractKernel):
            return SumKernel(kernels=[self, other])
        else:
            return SumKernel(kernels=[self, Constant(other)])

    def __radd__(self, other: Union["AbstractKernel", ScalarFloat]) -> "AbstractKernel":
        r"""Add two kernels together.
        Args:
            other (AbstractKernel): The kernel to be added to the current kernel.

        Returns
        -------
            AbstractKernel: A new kernel that is the sum of the two kernels.
        """
        return self.__add__(other)

    def __mul__(self, other: Union["AbstractKernel", ScalarFloat]) -> "AbstractKernel":
        r"""Multiply two kernels together.

        Args:
            other (AbstractKernel): The kernel to be multiplied with the current kernel.

        Returns
        -------
            AbstractKernel: A new kernel that is the product of the two kernels.
        """
        if isinstance(other, AbstractKernel):
            return ProductKernel(kernels=[self, other])
        else:
            return ProductKernel(kernels=[self, Constant(other)])

    @property
    def spectral_density(self) -> Optional[tfd.Distribution]:
        return None


@dataclass
class Constant(AbstractKernel):
    r"""
    A constant kernel. This kernel evaluates to a constant for all inputs.
    The scalar value itself can be treated as a model hyperparameter and learned during training.
    """

    constant: ScalarFloat = param_field(jnp.array(0.0))

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, " D"]): The left hand input of the kernel function.
            y (Float[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        return self.constant.squeeze()


@dataclass
class CombinationKernel(AbstractKernel):
    r"""A base class for products or sums of MeanFunctions."""

    kernels: List[AbstractKernel] = None
    operator: Callable = static_field(None)

    def __post_init__(self):
        # Add kernels to a list, flattening out instances of this class therein, as in GPFlow kernels.
        kernels_list: List[AbstractKernel] = []

        for kernel in self.kernels:
            if not isinstance(kernel, AbstractKernel):
                raise TypeError("can only combine Kernel instances")  # pragma: no cover

            if isinstance(kernel, self.__class__):
                kernels_list.extend(kernel.kernels)
            else:
                kernels_list.append(kernel)

        self.kernels = kernels_list

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, " D"]): The left hand input of the kernel function.
            y (Float[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        return self.operator(jnp.stack([k(x, y) for k in self.kernels]))


SumKernel = partial(CombinationKernel, operator=jnp.sum)
ProductKernel = partial(CombinationKernel, operator=jnp.prod)


@dataclass()
class AdditiveKernel(AbstractKernel):
    r"""Build an additive kernel from a list of individual base kernels for a specific maximum interaction depth."""

    kernels: list[AbstractKernel] = None
    max_interaction_depth: ScalarInt = static_field(1)
    interaction_variances: Float[Array, " p"] = param_field(jnp.array([1.0, 1.0]), bijector=tfb.Softplus())
    zeroth_order: bool = static_field(True)
    name: str = "AdditiveKernel"

    def __post_init__(self): # jax/jit requires specifying max_interaction depth even though this could be inferred from length of interaction_variances
        if self.zeroth_order:
            if not self.max_interaction_depth == len(self.interaction_variances) - 1:
                raise ValueError("Number of interaction variances must be equal to max_interaction_depth + 1")
        else:
            if not self.max_interaction_depth == len(self.interaction_variances):
                raise ValueError("Number of interaction variances must be equal to max_interaction_depth")


    def __call__(self, x: Num[Array, " D"], y: Num[Array, " D"]) -> ScalarFloat:
        r"""Compute the additive kernel between a pair of inputs.

        Args:
            x (Float[Array, " D"]): The left hand input of the kernel function.
            y (Float[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        x_sliced, y_sliced = self.slice_input(x), self.slice_input(y)
        ks = jnp.stack([k(x_sliced, y_sliced) for k in self.kernels]) # individual kernel evals


        # assert self.max_interaction_depth == 2
        # e_1 = self._compute_additive_terms_girad_newton(ks)
        # ks_05 = jnp.stack([k(x_sliced / jnp.sqrt(2.0), y_sliced / jnp.sqrt(2.0)) for k in self.kernels]) # individual kernel evals
        # e_05 = self._compute_additive_terms_girad_newton(ks_05)
        # return self.interaction_variances[0] * e_1[0] + self.interaction_variances[1] *e_1[1] + self.interaction_variances[2] *e_05[2]
        if self.zeroth_order:
            return jnp.sum(self._compute_additive_terms_girad_newton(ks) * self.interaction_variances)
        else:
            return jnp.sum(self._compute_additive_terms_girad_newton(ks)[1:] * self.interaction_variances)
        #return self._compute_additive_terms_girad_newton(ks) # combined kernel
            

    @jax.jit   
    def _compute_additive_terms_girad_newton(self, ks: Num[Array, " D"]) -> Num[Array, " p"]:
        r"""Given a list of inputs, compute a new list containing all products up to order
        `max_interaction_depth`. For efficiency, we us the Girad Newton identity 
        (i.e. O(d^2) instead of exponential).

        Args:
            ks (Num[Array, " D"]): The evaluations of the individual kernels.

        Returns
        -------
            ScalarFloat: The sum of products of individual kernels for each required order.
        """
        powers = jnp.arange(self.max_interaction_depth + 1)[:, None] # [p + 1, 1]
        s = jnp.power(ks[None, :],powers) # [p + 1, d]
        e = jnp.ones(shape=(self.max_interaction_depth+1), dtype=jnp.float64) # lazy init then populate
        for n in range(1, self.max_interaction_depth + 1): # has to be for loop because iterative
            thing = jax.vmap(lambda k: ((-1.0)**(k-1))*e[n-k]*s[k, :])(jnp.arange(1, n+1))
            e = e.at[n].set((1.0/n) *jnp.sum(thing))
        return e




    # @jax.jit
    # def _compute_additive_terms_girad_newton(self, ks: Num[Array, " D"]) -> Num[Array, " 1"]:
    #     r"""Given a list of inputs, compute a new list containing all products up to order
    #     `max_interaction_depth`. For efficiency, we us the Girad Newton identity 
    #     (i.e. O(d^2) instead of exponential).

    #     Args:
    #         ks (Num[Array, " D"]): The evaluations of the individual kernels.

    #     Returns
    #     -------
    #         ScalarFloat: The sum of products of individual kernels for each required order. todo
    #     """
    
    #     def do_recursion(ks_vec: Num[Array, "M D"]) -> Num[Array, "p M"]:
    #         powers = jnp.arange(self.max_interaction_depth+1)[:, None, None] # [p+1 , 1, 1]
    #         s = jnp.power(ks_vec,powers) # [p+1, M, d]
    #         e = jnp.ones(shape=(self.max_interaction_depth+1, jnp.shape(ks_vec)[0]), dtype=jnp.float64) # lazy init then populate [p+1, M]
    #         for n in range(1, self.max_interaction_depth+1 ): # has to be for loop because iterative (could unroll to speed up for worse mem)
    #             thing = jax.vmap(lambda k: ((-1.0)**(k-1))*e[n-k,:]*jnp.sum(s[k, :, :], -1))(jnp.arange(1, n+1)) # [n, M, d]
    #             e = e.at[n].set((1.0/n) *jnp.sum(thing, (0)))
    #         return e
    

    #     from jax import custom_jvp
    #     @custom_jvp 
    #     def calc_hd(ks: Num[Array, " D"], interaction_variances: Float[Array, " p"] ) -> Num[Array, " 1"]:
    #         return jnp.sum(do_recursion(ks[None,:])[:,0] * interaction_variances) # [p + 1]
            
    #     @calc_hd.defjvp
    #     def calc_hd_jvp(primals, tangents):
    #         ks , iv = primals
    #         ks_dot, iv_dot = tangents
        
    #         ks_vec = jnp.repeat(ks[None,:],jnp.shape(ks)[0], 0) - jnp.diag(ks) # [d, d]
    #         e = do_recursion(jnp.vstack([ks_vec, ks[None,:]])) # [p+1, d+1]
    #         return jnp.sum(e[:,-1] * iv),  jnp.sum(jnp.sum(e[:-1,:-1] * iv[:, None][1:],0) * ks_dot) + jnp.sum(e[:,-1] * iv_dot)

    #     return calc_hd(ks, self.interaction_variances)


    def get_specific_kernel(self, component_list: List[int] = []) -> AbstractKernel:
        r""" Get a specific kernel from the additive kernel corresponding to component_list.

        For example, requesting component_list = [0, 1] will return the kernel corresponding to the
        product of the first two kernels considered in additive kernel.

        Args:
            component_list (List[int]): The list representing the desired sub-kernel combination.

        Returns
        -------
            AbstractKernel: The individual kernel corresponding to the desired sub-kernel combination.
        
        
        """
        var = self.interaction_variances[len(component_list)]
        kernel = Constant(constant = var)
        for i in component_list:
            kernel = kernel * self.kernels[i]
        return kernel
    
