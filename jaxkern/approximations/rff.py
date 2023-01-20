from ..base import AbstractKernel
from ..computations import BasisFunctionComputation
from jax.random import KeyArray
from typing import Dict, Any


class RFF(AbstractKernel):
    def __init__(self, base_kernel: AbstractKernel, num_basis_fns: int) -> None:
        self._check_valid_base_kernel(base_kernel)
        self.base_kernel = base_kernel
        self.num_basis_fns = num_basis_fns
        # Set the computation engine to be basis function computation engine
        self.compute_engine = BasisFunctionComputation
        # Inform the compute engine of the number of basis functions
        self.compute_engine.num_basis_fns = num_basis_fns

    def init_params(self, key: KeyArray) -> Dict:
        base_params = self.base_kernel.init_params(key)
        n_dims = base_params["lengthscale"].shape[0]
        frequencies = self.base_kernel.spectral_density.sample(
            seed=key, sample_shape=(self.num_basis_fns, n_dims)
        )
        base_params["frequencies"] = frequencies
        return base_params

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def _check_valid_base_kernel(self, kernel):
        error_msg = """
        Base kernel must have a spectral density. Currently, only Matérn
        and RBF kernels have implemented spectral densities.
        """
        if kernel.spectral_density is None:
            raise ValueError(error_msg)
