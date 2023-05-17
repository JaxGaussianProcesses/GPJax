# 🔪 The sharp bits

## Pseudo-randomness

Libraries like Numpy and Scipy use *stateful* pseudorandom number generators (PRNGs).
However, the PRNG in JAX is stateless. This means that for a given function, the
return always return the same result unless the seed is changed. This is a good thing,
but it means that we need to be careful when using JAX's PRNGs.

To examine what it means for a PRNG to be stateful, consider the following example:

```python
import numpy as np
import jax.random as jr
key = jr.PRNGKey(123)

# Numpy
print('Numpy:')
print(np.random.random())
print(np.random.random())

print('\nJAX:')
print(jr.uniform(key))
print(jr.uniform(key))

print('\nSplitting key')
key, subkey = jr.split(key)
print(jr.uniform(subkey))
```
```console
Numpy:
0.5194454541172852
0.9815886617924413

JAX:
0.95821166
0.95821166

Splitting key
0.23886406
```
We can see that, in libraries like Numpy, the PRNG key's state is incremented whenever
a pseudorandom call is made. This can make debugging difficult to manage as it is not
always clear when a PRNG is being used. In JAX, the PRNG key is not incremented, and
so the same key will always return the same result. This has further positive benefits
for reproducibility.

GPJax relies on JAX's PRNGs for all random number generation. Whilst we try wherever possible to handle the PRNG key's state for you, care must be taken when defining your own models and inference schemes to ensure that the PRNG key is handled correctly. The [JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers) has an excellent section on this.

## Bijectors

Parameters such as the kernel's lengthscale or variance have their support defined on
a constrained subset of the real-line. During gradient-based optimisation, as we
approach the set's boundary, it becomes possible that we could step outside of the
set's support and introduce a numerical and mathematical error into our model. For
example, consider the variance parameter $`\sigma^2`$, which we know must be strictly
positive. If at $`t^{\text{th}}`$ iterate, our current estimate of $`\sigma^2`$ was
0.03 and our derivative informed us that $`\sigma^2`$ should decrease, then if our
learning rate is greater is than 0.03, we would end up with a negative variance term.

A simple, but impractical solution, would be to use a tiny learning rate which would
reduce the possibility of stepping outside of the parameter's support. However, this
would be incredibly costly and does not eradicate the problem. An alternative solution
is to apply a functional mapping to the parameter that projects it from a constrained
subspace of the real-line onto the entire real-line. Here, gradient updates are
applied in the unconstrained parameter space before transforming the value back to the
original support of the parameters. Such a transformation is known as a bijection.


In GPJax, we supply bijective functions using [Tensorflow Probability](https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors).
In our [PyTrees doc](examples/pytrees.md) document, we detail how the user can define
their own bijectors and attach them to the parameter(s) of their model.

## Positive-definiteness

> "Symmetric positive definiteness is one of the highest accolades to which a matrix can aspire" - Nicholas Highman, Accuracy and stability of numerical algorithms

### Why is positive-definiteness important?

The covariance matrix of a kernel is a symmetric positive definite matrix. As such, we
have a range of tools at our disposal to make subsequent operations on the covariance
matrix faster. One of these tools is the Cholesky factorisation that uniquely decomposes
any symmetric positive-definite matrix $`\mathbf{\Sigma}`$ by 
    
```math 
\begin{align}
    \mathbf{\Sigma} = \mathbf{L}\mathbf{L}^{\top}
\end{align}
```

where $`\mathbf{L}`$ is a lower triangular matrix. 

We make use of this result in GPJax when solving linear systems of equations of the
form $`\mathbf{A}\boldsymbol{x} = \boldsymbol{b}`$. Whilst seemingly abstract at first,
such problems are frequently encountered when constructing Gaussian process models. One
such example is frequently encountered in the regression setting for learning Gaussian
process Kernel hyperparameters. Here we have labels 
$`\boldsymbol{y} \sim \mathcal{N}(f(\boldsymbol{x}), \mathbf{\Sigma})`$ with $`f(\boldsymbol{x}) \sim \mathcal{N}(\boldsymbol{0}, \mathbf{K}_{\boldsymbol{xx}})`$ arising from zero-mean
Gaussian process prior and gram matrix $`\mathbf{K}_{\boldsymbol{xx}}`$ at the inputs
$`\boldsymbol{x}`$. Here the marginal log-likelihood comprises the following form

```math
\begin{align}
    \log p(\boldsymbol{y}) = 0.5\left(-\boldsymbol{y}^{\top}\left(\mathbf{K}_{\boldsymbol{xx}} + \sigma^2\mathbf{I} \right)^{-1}\boldsymbol{y} -\log\lvert \mathbf{K}_{\boldsymbol{xx}} + \mathbf{\Sigma}\rvert -n\log(2\pi)\right) ,
\end{align}
```

and the goal of inference is to maximise kernel hyperparameters (contained in the gram
matrix $`\mathbf{K}_{\boldsymbol{xx}}`$) and likelihood hyperparameters (contained in the
noise covariance $`\mathbf{\Sigma}`$). Computing the marginal log-likelihood (and its
gradients), draws our attention to the term

```math
\begin{align}
    \underbrace{\left(\mathbf{K}_{\boldsymbol{xx}} + \sigma^2\mathbf{I} \right)^{-1}}_{\mathbf{A}}\boldsymbol{y},
\end{align}
```

then we can see a solution can be obtained by solving the corresponding system of
equations. By working with $`\mathbf{L} = \operatorname{chol}{\mathbf{A}}`$ instead of
$`\mathbf{A}`$, we save a significant amount of floating-point operations (flops) by
solving two triangular systems of equations (one for $`\mathbf{L}`$ and another for
$`\mathbf{L}^{\top}`$) instead of one dense system of equations. Solving two triangular systems
of equations has complexity $`\mathcal{O}(n^3/6)`$; a vast improvement compared to
regular solvers that have $`\mathcal{O}(n^3)`$ complexity in the number of datapoints
$`n`$.

### The Cholesky drawback

While the computational acceleration provided by using Cholesky factors instead of dense
matrices is hopefully now apparent, an awkward numerical instability _gotcha_ can arise
due to floating-point rounding errors. When we evaluate a covariance function on a set
of points that are very _close_ to one another, eigenvalues of the corresponding
covariance matrix can get very small. So small that after numerical rounding, the
smallest eigenvalues can become negative-valued. While not truly less than zero, our
computer thinks they are, which becomes a problem when we want to compute a Cholesky
factor since this requires that the input matrix is positive-definite. If there are
negative eigenvalues, then this stipulation has been invalidated.

To resolve this, we apply some numerical _jitter_ to the diagonals of any Gram matrix.
Typically this is incredibly small, with $`10^{-6}`$ being the system default. However,
for some problems, this amount may need to be increased. 

## Slow-to-evaluate

Famously, a regular Gaussian process model (as detailed in 
[our regression notebook](examples/regression.py)) will scale cubically in the number of data points.
Consequently, if you try to fit your Gaussian process model to data set containing more
than several thousand data points, then you will likely incur a significant
computational overhead. In such cases, we recommend using Sparse Gaussian processes to
alleviate this issue.

Approximately, when the data contains less than 50000 data points, we recommend using
the uncollapsed evidence lower bound objective [@titsias2009] to optimise the parameters
of your sparse Gaussian process model. Such a model will scale linearly in the number of
data points and quadratically in the number of inducing points. We demonstrate its use
in [our sparse regression notebook](examples/collapsed_vi.py).

For data sets exceeding 50000 data points, even the sparse Gaussian process outlined
above will become computationally infeasible. In such cases, we recommend using the
collapsed evidence lower bound objective [@hensman2013gaussian] that allows stochastic
mini-batch optimisation of the parameters of your sparse Gaussian process model. Such a
model will scale linearly in the batch size and quadratically in the number of inducing
points. We demonstrate its use in 
[our sparse stochastic variational inference notebook](examples/uncollapsed_vi.py)