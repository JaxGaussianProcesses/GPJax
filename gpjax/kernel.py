import jax.numpy as jnp
from objax import Module
from typing import Callable, Optional
from jax import vmap
from .parameters import Parameter


class Kernel(Module):
    """
    Base class for all kernel functions. By inheriting the `Module` class from Objax, seamless interaction with model parameters is provided.
    """
    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: Optional naming of the kernel.
        """
        self.name = name
        self.spectral = False

    @staticmethod
    def gram(func: Callable, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the kernel's gram matrix given two, possibly identical, Jax arrays.

        Args:
            func: The kernel function to be called for any two values in x and y.
            x: An NxD vector of inputs.
            y: An MXE vector of inputs.

        Returns:
            An NxM gram matrix.
        """
        mapx1 = vmap(lambda x, y: func(x=x, y=y),
                     in_axes=(0, None),
                     out_axes=0)
        mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
        return mapx2(x, y)

    @staticmethod
    def dist(x: jnp.array, y: jnp.array, power: int) -> jnp.ndarray:
        """
        Compute the squared distance matrix between two inputs.

        Args:
            x: A 1xN vector
            y: A 1xM vector

        Returns:
            A float value quantifying the distance between x and y.
        """
        return jnp.sum((x - y)**power)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray):
        raise NotImplementedError


class Stationary(Kernel):
    """
    A base class for stationary kernels. That is, kernels whereby the Gram matrix's values are invariant to the value of the inputs, and instead depend only on the distance between the inputs.
    """
    def __init__(self,
                 lengthscale: Optional[jnp.ndarray] = jnp.array([1.]),
                 variance: Optional[jnp.ndarray] = jnp.array([1.]),
                 name: Optional[str] = "Stationary"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The inital value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(name=name)
        self.lengthscale = Parameter(lengthscale)
        self.variance = Parameter(variance)

    def scaled_distance(self, x: jnp.ndarray, y: jnp.ndarray, power: int) -> jnp.ndarray:
        ell = self.lengthscale.untransform
        tau = self.dist(x / ell, y / ell, power=power)
        return tau

    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return self.gram(self.kernel_func, X, Y).squeeze()

    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class RBF(Stationary):
    """
    The radial basis function kernel.
    """
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "RBF"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The initial value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(lengthscale=lengthscale, variance=variance, name=name)

    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the RBF specific function :math:`k(x,y)=\sigma^2 \exp\left( \frac{-0.5 \tau}{2\ell^2}\right) ` where  :math:`\tau = \lVert x-y \rVert_{2}^{2}`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=2)
        return sigma * jnp.exp(-0.5 * tau)


class Matern12(Stationary):
    """
    The Matern kernel with a smoothness parameter of 1/2.
    """
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "Matern 1/2"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The initial    value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(lengthscale=lengthscale, variance=variance, name=name)

    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the Matern 1/2 specific function :math:`k(x,y)=\sigma^2 \exp\left( \frac{-\tau}{\ell}\right) ` where  :math:`\tau = \lvert x-y \rvert`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=1)
        return sigma * jnp.exp(-0.5 * tau)


class Matern32(Stationary):
    """
    The Matern kernel with a smoothness parameter of 3/2.
    """
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "Matern 3/2"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The initial    value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(lengthscale=lengthscale, variance=variance, name=name)

    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the Matern 1/2 specific function :math:`k(x,y)=\sigma^2 (1 + \frac{\sqrt{3} \tau}{\ell}) \exp\left( \frac{\sqrt{3} \tau}{\ell}\right) ` where  :math:`\tau = \lvert x-y \rvert`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=1)
        return sigma*(1+(jnp.sqrt(3.)*tau))*jnp.exp(-jnp.sqrt(3.)*tau)


# class Matern52(Stationary):
#     """
#     The Matern kernel with a smoothness parameter of 5/2.
#     """
#     def __init__(self,
#                  lengthscale: jnp.ndarray = jnp.array([1.]),
#                  variance: jnp.ndarray = jnp.array([1.]),
#                  name: str = "Matern 5/2"):
#         """
#         Args:
#             lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
#             variance: The initial    value of the kernel's variance. This value controls the kernel's vertical amplitude.
#             name: Optional argument to name the kernel.
#         """
#         super().__init__(lengthscale=lengthscale, variance=variance, name=name)
#
#     def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
#         r"""
#         Compute the Matern 1/2 specific function :math:`k(x,y)=\sigma^2 (1 + \frac{\sqrt{5} \tau}{\ell} + \frac{2.5*\tau**2}{\ell**2}) \exp\left( \frac{\sqrt{5} \tau}{\ell}\right) ` where  :math:`\tau = \lvert x-y \rvert`.
#         """
#         sigma = self.variance.untransform
#         tau = self.scaled_distance(x, y, power=1)
#         return sigma*(1.0+jnp.sqrt(5.0)*tau + 5.0/3.0 * jnp.square(tau))*jnp.exp(-jnp.sqrt(5.0)*tau)
