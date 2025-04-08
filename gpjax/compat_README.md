# Compatibility Layer for Linear Algebra Operations

This directory contains a compatibility layer (`compat.py`) that provides JAX-based implementations of linear algebra operations that match the API of the `cola` library. This allows for easy switching between the two implementations by only changing imports.

## Current Implementation

The current implementation uses JAX-based functions for all linear algebra operations. This eliminates the dependency on the `cola` library while maintaining the same API.

## Switching Back to Cola

If you want to switch back to using the `cola` library in the future, follow these steps:

1. Add `cola-ml` back to the dependencies in `pyproject.toml`:
   ```toml
   dependencies = [
     # ... other dependencies ...
     "cola-ml==0.0.5",  # or the latest version
   ]
   ```

2. Modify `gpjax/compat.py` to re-export the cola functions instead of implementing them with JAX. Here's a template for how to do this:

   ```python
   """Compatibility layer for linear algebra operations.

   This module re-exports the cola library functions to maintain API compatibility.
   """

   # Re-export cola classes and functions
   from cola.annotations import PSD
   from cola.fns import dispatch
   from cola.linalg.algorithm_base import Algorithm, Auto
   from cola.linalg.decompositions.decompositions import Cholesky
   from cola.linalg.inverse.cg import CG
   from cola.linalg.inverse.inv import inv, solve
   from cola.linalg.trace.diag_trace import diag
   from cola.ops.operator_base import LinearOperator
   from cola.ops.operators import (
       BlockDiag,
       Dense,
       Diagonal,
       I_like,
       Identity,
       Kronecker,
       Triangular,
   )

   # Re-export logdet function with the same signature as our JAX implementation
   from cola.linalg.logdet.logdet import logdet as cola_logdet

   def logdet(
       A, algorithm1=None, algorithm2=None
   ):
       """Compute the log determinant of a matrix.
       
       This is a wrapper around cola's logdet function to maintain API compatibility.
       """
       return cola_logdet(A, algorithm1)
   ```

3. Run your tests to ensure everything works as expected.

## Benefits of This Approach

- **Minimal Code Changes**: When switching between implementations, you only need to modify `compat.py`, not all the files that use these functions.
- **Consistent API**: The API remains the same regardless of which implementation is used.
- **Flexibility**: You can easily switch between implementations for testing or performance reasons. 