# Design Principles

`GPJax` is designed to be a Gaussian process package that provides an
accurate representation of the underlying maths. Variable names are chosen to closely match the notation in {cite}`rasmussen2006gaussian`.
We here list the notation used in `GPJax` with its corresponding mathematical quantity.

## Gaussian process notation

| On paper                                    | GPJax code | Description                                                                     |
| ------------------------------------------- | ---------- | ------------------------------------------------------------------------------- |
| $n$                                         | n          | Number of train inputs                                                          |
| $\boldsymbol{x} = (x_1,\dotsc,x_{n})$       | x          | Train inputs                                                                    |
| $\boldsymbol{y} = (y_1,\dotsc,y_{n})$       | y          | Train labels                                                                    |
| $\boldsymbol{t}$                            | t          | Test inputs                                                                     |
| $f(\cdot)$                                  | f          | Latent function modelled as a GP                                                |
| $f({\boldsymbol{x}})$                       | fx         | Latent function at inputs $\boldsymbol{x}$                                      |
| $\boldsymbol{\mu}_{\boldsymbol{x}}$         | μx         | Prior mean at inputs $\boldsymbol{x}$                                           |
| $\mathbf{K}_{\boldsymbol{x}\boldsymbol{x}}$ | Kxx        | Kernel Gram matrix at inputs $\boldsymbol{x}$                                   |
| $\mathbf{L}_{\boldsymbol{x}}$               | Lx         | Lower Cholesky decomposition of $\boldsymbol{K}_{\boldsymbol{x}\boldsymbol{x}}$ |
| $\mathbf{K}_{\boldsymbol{t}\boldsymbol{x}}$ | Ktx        | Cross-covariance between inputs $\boldsymbol{t}$ and $\boldsymbol{x}$           |

## Sparse Gaussian process notation

| On paper                              | GPJax code | Description               |
| ------------------------------------- | ---------- | ------------------------- |
| $m$                                   | m          | Number of inducing inputs |
| $\boldsymbol{z} = (z_1,\dotsc,z_{m})$ | z          | Inducing inputs           |
| $\boldsymbol{u} = (u_1,\dotsc,u_{m})$ | u          | Inducing outputs          |
