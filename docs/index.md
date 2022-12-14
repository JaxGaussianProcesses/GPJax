# Welcome to GPJax!

GPJax is a didactic Gaussian process library that supports GPU
acceleration and just-in-time compilation. We seek to provide a flexible
API as close as possible to how the underlying mathematics is written on
paper to enable researchers to rapidly prototype and develop new ideas.

![Gaussian process posterior.](./_static/GP.svg)

You can view the source code for GPJax [here on
Github](https://github.com/thomaspinder/GPJax).

## \'Hello World\' example

Defining a Gaussian process posterior is as simple as typing the maths we
would write on paper. To see this, consider the following example.

````{tabs}
```{code-tab} py
import gpjax as gpx

kernel = gpx.kernels.RBF()
prior = gpx.gps.Prior(kernel = kernel)

likelihood = gpx.likelihoods.Gaussian(num_datapoints = 123)

posterior = prior * likelihood

```
```{tab} Maths
$$
\begin{align}
k(\cdot, \cdot') & = \sigma^2\exp\left(-\frac{\lVert \cdot- \cdot'\rVert_2^2}{2\ell^2}\right)\\
p(f(\cdot)) & = \mathcal{GP}(\mathbf{0}, k(\cdot, \cdot')) \\
p(y\,|\, f(\cdot)) & = \mathcal{N}(y\,|\, f(\cdot), \sigma_n^2) \\
p(f(\cdot) \,|\, y) & \propto p(f(\cdot))p(y\,|\, f(\cdot))\,.
\end{align}
$$
```
````

:::{note}
If you're new to Gaussian processes and want a gentle introduction, we have put together an introduction to GPs notebook that starts from Bayes' theorem and univariate Gaussian random variables. The notebook is linked [here](https://gpjax.readthedocs.io/en/latest/examples/intro_to_gps.html).
:::

:::{seealso}
To learn more, checkout the [regression
notebook](https://gpjax.readthedocs.io/en/latest/examples/regression.html).
:::

---

```{toctree}
---
maxdepth: 1
caption: Getting Started
hidden:
---
installation
design
contributing
examples/intro_to_gps
```

```{toctree}
---
maxdepth: 1
caption: Examples
hidden:
---
examples/regression
examples/classification
examples/uncollapsed_vi
examples/collapsed_vi
examples/graph_kernels
examples/barycentres
examples/haiku
examples/tfp_integration
```

```{toctree}
---
maxdepth: 1
caption: Guides
hidden:
---
examples/kernels
examples/yacht
```

```{toctree}
---
maxdepth: 1
caption: Experimental
hidden:
---
examples/natgrads
examples/geometric_kernels
```

```{toctree}
---
maxdepth: 1
caption: Package Reference
hidden:
---
api
```

# Bibliography

```{bibliography}
---
cited:
---
```
