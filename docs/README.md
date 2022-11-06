# Where to find the docs

The GPJax documentation can be found here:
https://gpjax.readthedocs.io/en/latest/

# How to build the docs

1. Install the requirements using `pip install -r docs/requirements.txt`
2. Make sure `pandoc` is installed
3. Run the make script `make html`

The corresponding HTML files can then be found in `docs/_build/html/`.

# How to write code documentation

Our documentation it is written in ReStructuredText for Sphinx. This is a
meta-language that is compiled into online documentation. For more details see
[Sphinx's documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).
As a result, our docstrings adhere to a specific syntax that has to be kept in
mind. Below we provide some guidelines.

## How much information to put in a docstring

A docstring should be informative. If in doubt, then it is best to add more
information to a docstring than less. Many users will skim documentation, so
please ensure the opening sentence or two of a docstring contains the core
information. Adding examples and mathematical descriptions to documentation is
highly desirable.

We are making an active effort within GPJax to improve our documentation. If you
spot any areas where there is missing information within the existing
documentation, then please either raise an issue or 
[create a pull request](https://gpjax.readthedocs.io/en/latest/contributing.html).

## An example docstring

An example docstring that adheres the principles of GPJax is given below. 
The docstring contains a simple, snappy introduction with links to auxillary 
components. More detail is then provided in the form of a mathematical 
description and a code example. The docstring is concluded with a description
of the objects attributes with corresponding types.

```python
@dataclass
class Prior(AbstractPrior):
    """A Gaussian process prior object. The GP is parameterised by a
    `mean <https://gpjax.readthedocs.io/en/latest/api.html#module-gpjax.mean_functions>`_
    and `kernel <https://gpjax.readthedocs.io/en/latest/api.html#module-gpjax.kernels>`_ function.

    A Gaussian process prior parameterised by a mean function :math:`m(\\cdot)` and a kernel
    function :math:`k(\\cdot, \\cdot)` is given by

    .. math::

        p(f(\\cdot)) = \mathcal{GP}(m(\\cdot), k(\\cdot, \\cdot)).

    To invoke a ``Prior`` distribution, only a kernel function is required. By default,
    the mean function will be set to zero. In general, this assumption will be reasonable
    assuming the data being modelled has been centred.

    Example:
        >>> import gpjax as gpx
        >>>
        >>> kernel = gpx.kernels.RBF()
        >>> prior = gpx.Prior(kernel = kernel)

    Attributes:
        kernel (Kernel): The kernel function used to parameterise the prior.
        mean_function (MeanFunction): The mean function used to parameterise the prior. Defaults to zero.
        name (str): The name of the GP prior. Defaults to "GP prior".
    """

    kernel: Kernel
    mean_function: Optional[AbstractMeanFunction] = Zero()
    name: Optional[str] = "GP prior"
```

### Documentation syntax

A helpful cheatsheet for writing restructured text can be found 
[here](https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst). In addition to that, we adopt the following convention when documenting
`dataclass` objects.

*  Class attributes should be specified using the `Attributes:` tag.
*  Method argument should be specified using the `Args:` tags.
*  All attributes and arguments should have types.
