# Welcome to GPJax!

GPJax is a didactic Gaussian process library that supports GPU
acceleration and just-in-time compilation. We seek to provide a flexible
API as close as possible to how the underlying mathematics is written on
paper to enable researchers to rapidly prototype and develop new ideas.

![Gaussian process posterior.](./_static/GP.svg)

You can view the source code for GPJax [here on
Github](https://github.com/thomaspinder/GPJax).

## `Hello World` example

Defining a Gaussian process posterior is as simple as typing the maths we
would write on paper. To see this, consider the following example.

=== "Python"

    ``` py
    import gpjax as gpx

    meanf = gpx.Zero()
    kernel = gpx.kernels.RBF()
    prior = gpx.gps.Prior(mean_function = meanf, kernel = kernel)

    likelihood = gpx.likelihoods.Gaussian(num_datapoints = 123)

    posterior = prior * likelihood
    ```

=== "Maths"

    $$
    \begin{align}
    k(\cdot, \cdot') & = \sigma^2\exp\left(-\frac{\lVert \cdot- \cdot'\rVert_2^2}{2\ell^2}\right)\\
    p(f(\cdot)) & = \mathcal{GP}(\mathbf{0}, k(\cdot, \cdot')) \\
    p(y\,|\, f(\cdot)) & = \mathcal{N}(y\,|\, f(\cdot), \sigma_n^2) \\
    p(f(\cdot) \,|\, y) & \propto p(f(\cdot))p(y\,|\, f(\cdot))\,.
    \end{align}
    $$

!!! note

    If you're new to Gaussian processes and want a gentle introduction, we have put together an introduction to GPs notebook that starts from Bayes' theorem and univariate Gaussian random variables. The notebook is linked [here](https://gpjax.readthedocs.io/en/latest/examples/intro_to_gps.html).

!!! seealso

    To learn more, checkout the [regression
    notebook](https://gpjax.readthedocs.io/en/latest/examples/regression.html).

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
