# Welcome to GPJax!

GPJax is a didactic Gaussian process (GP) library in JAX, supporting GPU
acceleration and just-in-time compilation. We seek to provide a flexible
API to enable researchers to rapidly prototype and develop new ideas.

![Gaussian process posterior.](static/GP.svg)


## "Hello, GP!"

Typing GP models is as simple as the maths we
would write on paper, as shown below.

=== "Python"

    ``` py
    import gpjax as gpx

    mean = gpx.mean_functions.Zero()
    kernel = gpx.kernels.RBF()
    prior = gpx.gps.Prior(mean_function = mean, kernel = kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints = 123)

    posterior = prior * likelihood
    ```

=== "Math"

    ```math
    \begin{align}
    k(\cdot, \cdot') & = \sigma^2\exp\left(-\frac{\lVert \cdot- \cdot'\rVert_2^2}{2\ell^2}\right)\\
    p(f(\cdot)) & = \mathcal{GP}(\mathbf{0}, k(\cdot, \cdot')) \\
    p(y\,|\, f(\cdot)) & = \mathcal{N}(y\,|\, f(\cdot), \sigma_n^2) \\ \\
    p(f(\cdot) \,|\, y) & \propto p(f(\cdot))p(y\,|\, f(\cdot))\,.
    \end{align}
    ```

## Quick start

!!! Install

    GPJax can be installed via pip. See our [installation guide](installation.md) for further details.

    ```bash
    pip install gpjax
    ```

!!! New

    New to GPs? Then why not check out our [introductory notebook](_examples/intro_to_gps.md) that starts from Bayes' theorem and univariate Gaussian distributions.

!!! Begin

    Looking for a good place to start? Then why not begin with our [regression
    notebook](https://docs.jaxgaussianprocesses.com/examples/regression/).

## Citing GPJax

If you use GPJax in your research, please cite our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.04455#).

```
@article{Pinder2022,
  doi = {10.21105/joss.04455},
  url = {https://doi.org/10.21105/joss.04455},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {75},
  pages = {4455},
  author = {Thomas Pinder and Daniel Dodd},
  title = {GPJax: A Gaussian Process Framework in JAX},
  journal = {Journal of Open Source Software}
}
```
