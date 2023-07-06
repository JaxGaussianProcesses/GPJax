# Where to find the docs

The GPJax documentation can be found here: https://docs.jaxgaussianprocesses.com/

# How to build the docs

1. Ensure you have installed the requirements using `poetry install` in the root
   directory.
2. Make sure `pandoc` is installed
3. Run the command `poetry run mkdocs serve` in the root directory.

The documentation will then be served at an IP address printed, which can then be opened
in a browser of you choice e.g. `Serving on http://127.0.0.1:8000/`.

# How to write code documentation

Our documentation is generated using [MkDocs](https://www.mkdocs.org/). This
automatically creates online documentation from docstrings, with full support for
Markdown. Longer tutorial-style notebooks are also converted to webpages by MkDocs, with
these notebooks being stored in the `docs/examples` directory. If you write a new
notebook and wish to add it to the documentation website, add it to the `nav` section of
the `mkdocs.yml` file found in the root directory.

Below we provide some guidelines for writing docstrings.

## How much information to put in a docstring

A docstring should be informative. If in doubt, then it is best to add more information
to a docstring than less. Many users will skim documentation, so please ensure the
opening sentence or two of a docstring contains the core information. Adding examples
and mathematical descriptions to documentation is highly desirable.

We are making an active effort within GPJax to improve our documentation. If you spot
any areas where there is missing information within the existing documentation, then
please either raise an issue or [create a pull
request](https://docs.jaxgaussianprocesses.com/contributing/).

## An example docstring

An example docstring that adheres the principles of GPJax is given below. The docstring
contains a simple, snappy introduction with links to auxiliary components. More detail
is then provided in the form of a mathematical description and a code example. The
docstring is concluded with a description of the objects attributes with corresponding
types.

```python
from gpjax.gps import AbstractPrior
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.kernels import AbstractKernel
from typing import Optional

class Prior(AbstractPrior):
    r"""A Gaussian process prior object.

    The GP is parameterised by a
    [mean](https://docs.jaxgaussianprocesses.com/api/mean_functions/)
    and [kernel](https://docs.jaxgaussianprocesses.com/api/kernels/base/) function.

    A Gaussian process prior parameterised by a mean function $`m(\cdot)`$ and a kernel
    function $`k(\cdot, \cdot)`$ is given by
    $`p(f(\cdot)) = \mathcal{GP}(m(\cdot), k(\cdot, \cdot))`$.

    To invoke a `Prior` distribution, a kernel and mean function must be specified.

    Example:
        >>> import gpjax as gpx
        >>>
        >>> meanf = gpx.mean_functions.Zero()
        >>> kernel = gpx.kernels.RBF()
        >>> prior = gpx.Prior(mean_function=meanf, kernel = kernel)

    Attributes:
        kernel (Kernel): The kernel function used to parameterise the prior.
        mean_function (MeanFunction): The mean function used to parameterise the prior. Defaults to zero.
        name (str): The name of the GP prior. Defaults to "GP prior".
    """

    kernel: AbstractKernel
    mean_function: AbstractMeanFunction
    name: Optional[str] = "GP prior"
```

### Documentation syntax

We adopt the following convention when documenting objects:

*  Class attributes should be specified using the `Attributes:` tag.
*  Method argument should be specified using the `Args:` tags.
*  Values returned by a method should be specified using the `Returns:` tag.
*  All attributes, arguments and returned values should have types.

!!! attention "Note"

    Inline math in docstrings needs to be rendered within both `$` and `` symbols to be correctly rendered by MkDocs. For instance, where one would typically write `$k(x,y)$` in standard LaTeX, in docstrings you are required to write ``$`k(x,y)`$`` in order for the math to be correctly rendered by MkDocs.
