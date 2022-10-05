# Welcome to GPJax!

GPJax is a didactic Gaussian process library that supports GPU
acceleration and just-in-time compilation. We seek to provide a flexible
API as close as possible to how the underlying mathematics is written on
paper to enable researchers to rapidly prototype and develop new ideas.

![Gaussian process posterior.](./_static/GP.svg)

You can view the source code for GPJax [here on
Github](https://github.com/thomaspinder/GPJax).

## \'Hello World\' example

Defining a Gaussian process posterior is simple as typing the maths we
would write on paper.

```python
import gpjax as gpx

kernel = gpx.kernels.RBF()
prior = gpx.gps.Prior(kernel = kernel)

likelihood = gpx.likelihoods.Gaussian(num_datapoints = 123)

posterior = prior * likelihood
```

For comparison, the corresponding model could be written as

$$
\begin{align}
k(\cdot, \cdot') & = \sigma^2\exp\left(-\frac{\lVert \cdot- \cdot'\rVert_2^2}{2\ell^2}\right)\\
p(f(\cdot)) & = \mathcal{GP}(\mathbf{0}, k(\cdot, \cdot')) \\
p(y\,|\, f(\cdot)) & = \mathcal{N}(y\,|\, f(\cdot), \sigma_n^2) \\
p(f(\cdot) \,|\, y) & \propto p(f(\cdot))p(y\,|\, f(\cdot))\,.
\end{align}
$$

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
intro_to_gps
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
