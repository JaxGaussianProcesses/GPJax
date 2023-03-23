from ..base import AbstractKernel
from ..computations import BasisFunctionComputation
from jax.random import KeyArray
from typing import Dict, Any


class RFF(AbstractKernel):
    """Computes an approximation of the kernel using Random Fourier Features.

    All stationary kernels are equivalent to the Fourier transform of a probability
    distribution. We call the corresponding distribution the spectral density. Using
    a finite number of basis functions, we can compute the spectral density using a
    Monte-Carlo approximation. This is done by sampling from the spectral density and
    computing the Fourier transform of the samples. The kernel is then approximated by
    the inner product of the Fourier transform of the samples with the Fourier
    transform of the data.

    The key reference for this implementation is the following papers:
    - 'Random Features for Large-Scale Kernel Machines' by Rahimi and Recht (2008).
    - 'On the Error of Random Fourier Features' by Sutherland and Schneider (2015).

    Args:
        AbstractKernel (_type_): _description_
    """

    def __init__(self, base_kernel: AbstractKernel, num_basis_fns: int) -> None:
        """Initialise the Random Fourier Features approximation.

        Args:
            base_kernel (AbstractKernel): The kernel that is to be approximated. This kernel must be stationary.
            num_basis_fns (int): The number of basis functions that should be used to approximate the kernel.
        """
        self._check_valid_base_kernel(base_kernel)
        self.base_kernel = base_kernel
        self.num_basis_fns = num_basis_fns
        # Set the computation engine to be basis function computation engine
        self.compute_engine = BasisFunctionComputation
        # Inform the compute engine of the number of basis functions
        self.compute_engine.num_basis_fns = num_basis_fns

    def init_params(self, key: KeyArray) -> Dict:
        """Initialise the parameters of the RFF approximation.

        Args:
            key (KeyArray): A pseudo-random number generator key.

        Returns:
            Dict: A dictionary containing the original kernel's parameters and the initial frequencies used in RFF approximation.
        """
        base_params = self.base_kernel.init_params(key)
        n_dims = self.base_kernel.ndims
        frequencies = self.base_kernel.spectral_density.sample(
            seed=key, sample_shape=(self.num_basis_fns, n_dims)
        )
        base_params["frequencies"] = frequencies
        return base_params

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def _check_valid_base_kernel(self, kernel: AbstractKernel):
        """Verify that the base kernel is valid for RFF approximation.

        Args:
            kernel (AbstractKernel): The kernel to be checked.
        """
        error_msg = """
        Base kernel must have a spectral density. Currently, only Matérn
        and RBF kernels have implemented spectral densities.
        """
        if kernel.spectral_density is None:
            raise ValueError(error_msg)
