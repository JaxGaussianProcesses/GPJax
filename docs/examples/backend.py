# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: gpjax
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Backend Module Design
#
# Since v0.9, GPJax is built upon Flax's [NNX](https://flax.readthedocs.io/en/latest/nnx/index.html) module. This transition allows for more efficient parameter handling, improved integration with Flax and Flax-based libraries, and enhanced flexibility in model design. This notebook provides a high-level overview of the backend module design in GPJax. For an introduction to NNX, please refer to the [official documentation](https://flax.readthedocs.io/en/latest/nnx/index.html).
#

# %%
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import gpjax as gpx

plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Parameters
#
# The biggest change bought about by the transition to an NNX backend is the increased support we now provide for handling parameters. As discussed in our [Sharp Bits - Bijectors Doc](https://docs.jaxgaussianprocesses.com/sharp_bits/#bijectors), GPJax uses bijectors to transform constrained parameters to unconstrained parameters during optimisation. You may now register the support of a parameter using our `Parameter` class. To see this, consider the constant mean function who contains a single constant parameter whose value ordinarily exists on the real line. We can register this parameter as follows:

# %%
from gpjax.mean_functions import Constant
from gpjax.parameters import Real

constant_param = Real(value=1.0)
meanf = Constant(constant_param)
meanf

# %% [markdown]
# However, suppose you wish your mean function's constant parameter to be strictly positive. This is easy to achieve by using the correct Parameter type.

# %%
from gpjax.parameters import PositiveReal

constant_param = PositiveReal(value=1.0)
meanf = Constant(constant_param)
meanf

# %% [markdown]
# Were we to try and instantiate the `PositiveReal` class with a negative value, then an explicit error would be raised.

# %%
try:
    PositiveReal(value=-1.0)
except ValueError as e:
    print(e)

# %% [markdown]
# ### Parameter Transforms
#
# With a parameter instantiated, you likely wish to transform the parameter's value from its constrained support onto the entire real line. To do this, you can apply the `transform` function to the parameter. To control the bijector used to transform the parameter, you may pass a set of bijectors into the transform function. Under-the-hood, the `transform` function is looking up the bijector of a parameter using it's `_tag` field in the bijector dictionary, and then applying the bijector to the parameter's value using a tree map operation.

# %%
print(constant_param._tag)

# %% [markdown]
# For most users, you will not need to worry about this as we provide a set of default bijectors that are defined for all the parameter types we support. However, see our [Kernel Guide Notebook](https://docs.jaxgaussianprocesses.com/examples/constructing_new_kernels/) to see how you can define your own bijectors and parameter types.

# %%
from gpjax.parameters import DEFAULT_BIJECTION, transform

print(DEFAULT_BIJECTION[constant_param._tag])

# %% [markdown]
# We see here that the Softplus bijector is specified as the default for strictly positive parameters. To apply this, we may invoke the following

# %%
transform(constant_param, DEFAULT_BIJECTION)

# %% [markdown]
# ### Transforming Multiple Parameters
#
# In the above, we transformed a single parameter. However, in practice your parameters may be nested within several functions e.g., a kernel function within a GP model. Fortunately, transforming several parameters is a simple operation that we here demonstrate for a regular GP poster

# %%
kernel = gpx.kernels.Matern32()
meanf = gpx.mean_functions.Constant()

prior = gpx.gps.Prior(meanf, kernel)


likelihood = gpx.likelihoods.Gaussian(100)
posterior = likelihood * prior

# %% [markdown]
# # Backend Module Design
#
# Since v0.9, GPJax is built upon Flax's [NNX](https://flax.readthedocs.io/en/latest/nnx/index.html) module. This transition allows for more efficient parameter handling, improved integration with Flax and Flax-based libraries, and enhanced flexibility in model design. This notebook provides a high-level overview of the backend module design in GPJax. For an introduction to NNX, please refer to the [official documentation](https://flax.readthedocs.io/en/latest/nnx/index.html).
#

# %% [markdown]
# ## NNX Modules
#
# To demonstrate the ease of use and flexibility of NNX modules, we will implement a linear mean function using the existing abstractions in GPJax. For inputs $x_n \in \mathbb{R}^d$, the linear mean function $m(x): \mathbb{R}^d \to \mathbb{R}$ is defined as:
# $$
# m(x) = \alpha + \sum_{i=1}^d \beta_i x_i
# $$
# where $\alpha \in \mathbb{R}$ and $\beta_i \in \mathbb{R}$ are the parameters of the mean function. Let's now implement that using the new NNX backend.

# %%
import typing as tp

from jaxtyping import Float, Num

from gpjax.mean_functions import AbstractMeanFunction
from gpjax.parameters import Real, Parameter
from gpjax.typing import ScalarFloat, Array


class LinearMeanFunction(AbstractMeanFunction):
    def __init__(
        self,
        intercept: tp.Union[ScalarFloat, Float[Array, " O"], Parameter] = 0.0,
        slope: tp.Union[ScalarFloat, Float[Array, " D O"], Parameter] = 0.0,
    ):
        if isinstance(intercept, Parameter):
            self.intercept = intercept
        else:
            self.intercept = Real(jnp.array(intercept))

        if isinstance(slope, Parameter):
            self.slope = slope
        else:
            self.slope = Real(jnp.array(slope))

    def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N O"]:
        return self.intercept.value + jnp.dot(x, self.slope.value)


# %% [markdown]
# As we can see, the implementation is straightforward and concise. The `AbstractMeanFunction` module is a subclass of `nnx.Module`. From here, we inform the module about the parameters
#

# %%
X = jnp.linspace(-5.0, 5.0, 100)[:, None]

meanf = LinearMeanFunction(intercept=1.0, slope=2.0)
plt.plot(X, meanf(X))

# %%
