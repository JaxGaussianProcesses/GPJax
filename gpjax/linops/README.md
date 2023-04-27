# LinOps

The `linops` submodule is a lightweight linear operator library written in [`jax`](https://github.com/google/jax).

# Overview
Consider solving a diagonal matrix $A$ against a vector $b$.

```python
import jax.numpy as jnp

n = 1000
diag = jnp.linspace(1.0, 2.0, n)

A = jnp.diag(diag)
b = jnp.linspace(3.0, 4.0, n)

# A⁻¹ b
jnp.solve(A, b)
```
Doing so is costly in large problems. Storing the matrix gives rise to memory costs of $O(n^2)$, and inverting the matrix costs $O(n^3)$ in the number of data points $n$.

But hold on a second. Notice:

- We only have to store the diagonal entries to determine the matrix $A$. Doing so, would reduce memory costs from $O(n^2)$ to $O(n)$.
- To invert $A$, we only need to take the reciprocal of the diagonal, reducing inversion costs from $O(n^3)$, to $O(n)$.

`JaxLinOp` is designed to exploit stucture of this kind.
```python
from gpjax import linops

A = linops.DiagonalLinearOperator(diag = diag)

# A⁻¹ b
A.solve(b)
```
`linops` is designed to automatically reduce cost savings in matrix addition, multiplication, computing log-determinants and more, for other matrix stuctures too!

# Custom Linear Operator (details to come soon)

The flexible design of `linops` will allow users to impliment their own custom linear operators.

```python
from gpjax.linops import LinearOperator

class MyLinearOperator(LinearOperator):

  def __init__(self, ...)
    ...

# There will be a minimal number methods that users need to impliment for their custom operator.
# For optimal efficiency, we'll make it easy for the user to add optional methods to their operator,
# if they give better performance than the defaults.
```
