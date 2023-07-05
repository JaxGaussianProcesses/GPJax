# %% [markdown]
# # Introduction to Kernels

# %% [markdown]
# In this guide we provide an introduction to kernels, and the role they play in Gaussian process models.

# %%
# Enable Float64 for more stable matrix inversions.
from jax.config import config

config.update("jax_enable_x64", True)

from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
from docs.examples.utils import clean_legend

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

key = jr.PRNGKey(42)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# Using Gaussian Processes (GPs) to model functions can offer several advantages over alternative methods, such as deep neural networks. One key advantage is their rich quantification of uncertainty; not only do they provide *point estimates* for the values taken by a function throughout its domain, but they provide a full predictive posterior *distribution* over the range of values the function may take. This rich quantification of uncertainty is useful in many applications, such as Bayesian optimisation, which relies on being able to make *uncertainty-aware* decisions.
#
# However, another advantage of GPs is the ability for one to place *priors* on the functions being modelled. For instance, one may know that the underlying function being modelled observes certain characteristics, such as being *periodic* or having a certain level of *smoothness*. The *kernel*, or *covariance function*, is the primary means through which one is able to encode such prior knowledge about the function being modelled. This enables one to equip the GP with inductive biases which enable it to learn from data more efficiently, whilst generalising to unseen data more effectively.
#
# In this notebook we'll develop some intuition for what kinds of priors are encoded through the use of different kernels, and how this can be useful when modelling different types of functions.

# %% [markdown]
# ## Introducing a Common Family of Kernels - The Matérn Family

# %% [markdown]
# Intuitively, the kernel defines the notion of *similarity* between the value taken at two points, $\mathbf{x}$ and $\mathbf{x}'$, by a function $f$, and will be denoted as $k(\mathbf{x}, \mathbf{x}')$:
#
# $$k(\mathbf{x}, \mathbf{x}') = \text{Cov}[f(\mathbf{x}), f(\mathbf{x}')]$$
#
#  One would expect that, given a previously unobserved test point $\mathbf{x}^*$, training points which are *closest* to this unobserved point will be most similar to it. As such, the kernel is used to define this notion of similarity within the GP framework. It tends to be up to the user to select a kernel which is appropriate for the function being modelled.
#
# One of the most widely used families of kernels is the Matérn family. These kernels take on the following form:
#
# $$k_{\nu}(\mathbf{x}, \mathbf{x'}) = \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)}\left(\sqrt{2\nu} \frac{|\mathbf{x} - \mathbf{x'}|}{\kappa}\right)^{\nu} K_{\nu} \left(\sqrt{2\nu} \frac{|\mathbf{x} - \mathbf{x'}|}{\kappa}\right)$$
#
# where $K_{\nu}$ is a modified Bessel function, $\nu$, $\kappa$ and $\sigma^2$ are hyperparameters specifying the mean-square differentiability, lengthscale and variability respectively, and $|\cdot|$ is used to denote the Euclidean norm.
#
# In the limit of $\nu \to \infty$ this yields the *squared-exponential*, or *radial basis function (RBF)*, kernel, which is infinitely mean-square differentiable:
#
# $$k_{\infty}(\mathbf{x}, \mathbf{x'}) = \sigma^2 \exp\left(-\frac{|\mathbf{x} - \mathbf{x'}|^2}{2\kappa^2}\right)$$
#
# But what kind of functions does this kernel encode prior knowledge about? Let's take a look at some samples from GP priors defined used Matérn kernels with different values of $\nu$:

# %%
kernels = [
    gpx.kernels.Matern12(),
    gpx.kernels.Matern32(),
    gpx.kernels.Matern52(),
    gpx.kernels.RBF(),
]
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(7, 6), tight_layout=True)

x = jnp.linspace(-3.0, 3.0, num=200).reshape(-1, 1)

meanf = gpx.mean_functions.Zero()

for k, ax in zip(kernels, axes.ravel()):
    prior = gpx.Prior(mean_function=meanf, kernel=k)
    rv = prior(x)
    y = rv.sample(seed=key, sample_shape=(10,))
    ax.plot(x, y.T, alpha=0.7)
    ax.set_title(k.name)


# %% [markdown]
# It should be noted that commonly used Matérn kernels use half-integer values of $\nu$, such as $\nu = 1/2$ or $\nu = 5/2$. The fraction is sometimes omitted when naming the kernel, so that $\nu = 1/2$ is referred to as the Matérn12 kernel, and $\nu = 5/2$ is referred to as the Matérn52 kernel.
#
# The plots above clearly show that the choice of $\nu$ has a large impact on the *smoothness* of the functions being modelled by the GP, with functions drawn from GPs defined with the Matérn kernel becoming increasingly smooth as $\nu \to \infty$. More formally, this notion of smoothness is captured through the mean-square differentiability of the function being modelled. Functions sampled from GPs using a Matérn kernel are $k$-times mean-square differentiable, if and only if $\nu > k$. For instance, functions sampled from a GP using a Matérn12 kernel are zero times mean-square differentiable, and functions sampled from a GP using the RBF kernel are infinitely mean-square differentiable.
#
# As an important aside, a general property of the Matérn family of kernels is that they are examples of *stationary* kernels. This means that they only depend on the *displacement* of the two points being compared, $\mathbf{x} - \mathbf{x}'$, and not on their absolute values. This is a useful property to have, as it means that the kernel is invariant to translations in the input space. They also go beyond this, as they only depend on the Euclidean *distance* between the two points being compared, $|\mathbf{x} - \mathbf{x}'|$. Kernels which satisfy this property are known as *isotropic* kernels. This makes the function invariant to all rigid motions in the input space, such as rotations.

# %% [markdown]
# ## Inferring Kernel Hyperparameters

# %% [markdown]
# Most kernels have several *hyperparameters*, which we denote $\mathbf{\theta}$, which encode different assumptions about the underlying function being modelled. For the Matérn family descibred above, $\mathbf{\theta} = \{\nu, \kappa, \sigma\}$. A fully Bayesian approach to dealing with hyperparameters would be to place a prior over them, and marginalise over the posterior derived from the data in order to perform predictions. However, this is often computationally very expensive, and so a common approach is to instead *optimise* the hyperparameters by maximising the log marginal likelihood of the data. Given training data $\mathbf{D} = (\mathbf{X}, \mathbf{y})$, assumed to contain some additive Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the log marginal likelihood of the dataset is defined as:
#
# $$ \begin{aligned}
# \log(p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta})) &= \log\left(\int p(\mathbf{y} | \mathbf{f}, \mathbf{X}, \boldsymbol{\theta}) p(\mathbf{f} | \mathbf{X}, \boldsymbol{\theta}) d\mathbf{f}\right) \nonumber \\
# &= - \frac{1}{2} \mathbf{y} ^ \top \left(K(\mathbf{X}, \mathbf{X}) + \sigma^2 \mathbf{I} \right)^{-1} \mathbf{y} - \frac{1}{2} \log |K(\mathbf{X}, \mathbf{X}) + \sigma^2 \mathbf{I}| - \frac{n}{2} \log 2 \pi
# \end{aligned}$$

# %% [markdown]
# We'll demonstrate the advantages of being able to infer kernel parameters from the training data by fitting a GP to the widely used [Forrester function](https://www.sfu.ca/~ssurjano/forretal08.html):
#
# $$f(x) = (6x - 2)^2 \sin(12x - 4)$$


# %%
# Forrester function
def forrester(x):
    return (6 * x - 2) ** 2 * jnp.sin(12 * x - 4)


n = 5

training_x = jr.uniform(key=key, minval=0, maxval=1, shape=(n,)).reshape(-1, 1)
training_y = forrester(training_x)
D = gpx.Dataset(X=training_x, y=training_y)

test_x = jnp.linspace(0, 1, 100).reshape(-1, 1)
test_y = forrester(test_x)

# %% [markdown]
# First we define our model, using the Matérn32 kernel, and construct our posterior *without* optimising the kernel hyperparameters:

# %%
mean = gpx.mean_functions.Zero()
kernel = gpx.kernels.Matern32(
    lengthscale=jnp.array(2.0)
)  # Initialise our kernel lengthscale to 2.0

prior = gpx.Prior(mean_function=mean, kernel=kernel)

likelihood = gpx.Gaussian(
    num_datapoints=D.n, obs_noise=jnp.array(1e-6)
)  # Our function is noise-free, so we set the observation noise to a very small value
likelihood = likelihood.replace_trainable(obs_noise=False)

no_opt_posterior = prior * likelihood

# %% [markdown]
# We can then optimise the hyperparmeters by minimising the negative log marginal likelihood of the data:

# %%
negative_mll = gpx.objectives.ConjugateMLL(negative=True)
negative_mll(no_opt_posterior, train_data=D)
negative_mll = jit(negative_mll)

opt_posterior, history = gpx.fit(
    model=no_opt_posterior,
    objective=negative_mll,
    train_data=D,
    optim=ox.adam(learning_rate=0.01),
    num_iters=2000,
    safe=True,
    key=key,
)


# %%
opt_latent_dist = opt_posterior.predict(test_x, train_data=D)
opt_predictive_dist = opt_posterior.likelihood(opt_latent_dist)

opt_predictive_mean = opt_predictive_dist.mean()
opt_predictive_std = opt_predictive_dist.stddev()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
ax1.plot(training_x, training_y, "x", label="Observations", color=cols[0], alpha=0.5)
ax1.fill_between(
    test_x.squeeze(),
    opt_predictive_mean - 2 * opt_predictive_std,
    opt_predictive_mean + 2 * opt_predictive_std,
    alpha=0.2,
    label="Two sigma",
    color=cols[1],
)
ax1.plot(
    test_x,
    opt_predictive_mean - 2 * opt_predictive_std,
    linestyle="--",
    linewidth=1,
    color=cols[1],
)
ax1.plot(
    test_x,
    opt_predictive_mean + 2 * opt_predictive_std,
    linestyle="--",
    linewidth=1,
    color=cols[1],
)
ax1.plot(
    test_x, test_y, label="Latent function", color=cols[0], linestyle="--", linewidth=2
)
ax1.plot(test_x, opt_predictive_mean, label="Predictive mean", color=cols[1])
ax1.set_title("Posterior with Hyperparameter Optimisation")
ax1.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))

no_opt_latent_dist = no_opt_posterior.predict(test_x, train_data=D)
no_opt_predictive_dist = no_opt_posterior.likelihood(no_opt_latent_dist)

no_opt_predictive_mean = no_opt_predictive_dist.mean()
no_opt_predictive_std = no_opt_predictive_dist.stddev()

ax2.plot(training_x, training_y, "x", label="Observations", color=cols[0], alpha=0.5)
ax2.fill_between(
    test_x.squeeze(),
    no_opt_predictive_mean - 2 * no_opt_predictive_std,
    no_opt_predictive_mean + 2 * no_opt_predictive_std,
    alpha=0.2,
    label="Two sigma",
    color=cols[1],
)
ax2.plot(
    test_x,
    no_opt_predictive_mean - 2 * no_opt_predictive_std,
    linestyle="--",
    linewidth=1,
    color=cols[1],
)
ax2.plot(
    test_x,
    no_opt_predictive_mean + 2 * no_opt_predictive_std,
    linestyle="--",
    linewidth=1,
    color=cols[1],
)
ax2.plot(
    test_x, test_y, label="Latent function", color=cols[0], linestyle="--", linewidth=2
)
ax2.plot(test_x, no_opt_predictive_mean, label="Predictive mean", color=cols[1])
ax2.set_title("Posterior without Hyperparameter Optimisation")
ax2.legend(loc="center left", bbox_to_anchor=(0.975, 0.5))

# %% [markdown]
# We can see that optimising the hyperparameters by minimising the negative log marginal likelihood of the data results in a more faithful fit of the GP to the data. In particular, we can observe that the GP using optimised hyperparameters is more accurately able to reflect uncertainty in its predictions, as opposed to the GP using the default parameters, which is overconfident in its predictions.
#
# The lengthscale, $\kappa$, and variance, $\sigma^2$, are shown below, both before and after optimisation:

# %%
no_opt_lengthscale = no_opt_posterior.prior.kernel.lengthscale
no_opt_variance = no_opt_posterior.prior.kernel.variance
opt_lengthscale = opt_posterior.prior.kernel.lengthscale
opt_variance = opt_posterior.prior.kernel.variance

print(f"Optimised Lengthscale: {opt_lengthscale} and Variance: {opt_variance}")
print(
    f"Non-Optimised Lengthscale: {no_opt_lengthscale} and Variance: {no_opt_variance}"
)

# %% [markdown]
# ## Expressing Other Priors with Different Kernels

# %% [markdown]
# Whilst the Matérn kernels are often used as a first choice of kernel, and they often perform well due to their smoothing properties often being well-aligned with the properties of the underlying function being modelled, sometimes more prior knowledge is known about the function being modelled. For instance, it may be known that the function being modelled is *periodic*. In this case, a suitable kernel choice would be the *periodic* kernel:
#
# $$k(\mathbf{x}, \mathbf{x}') = \sigma^2 \exp \left( -\frac{1}{2} \sum_{i=1}^{D} \left(\frac{\sin (\pi (\mathbf{x}_i - \mathbf{x}_i')/p)}{\ell}\right)^2 \right)$$
#
# with $D$ being the dimensionality of the inputs.
#
# Below we show $10$ samples drawn from a GP prior using the periodic kernel:

# %%
mean = gpx.mean_functions.Zero()
kernel = gpx.kernels.Periodic()
prior = gpx.Prior(mean_function=mean, kernel=kernel)

x = jnp.linspace(-3.0, 3.0, num=200).reshape(-1, 1)
rv = prior(x)
y = rv.sample(seed=key, sample_shape=(10,))

fig, ax = plt.subplots()
ax.plot(x, y.T, alpha=0.7)
ax.set_title("Samples from the Periodic Kernel")
plt.show()

# %% [markdown]
# In other scenarios, it may be known that the underlying function is *linear*, in which case the *linear* kernel would be a suitable choice:
#
# $$k(\mathbf{x}, \mathbf{x}') = \sigma^2 \mathbf{x}^\top \mathbf{x}'$$
#
# Unlike the kernels shown above, the linear kernel is *not* stationary, and so it is not invariant to translations in the input space.
#
# Below we show $10$ samples drawn from a GP prior using the linear kernel:

# %%
mean = gpx.mean_functions.Zero()
kernel = gpx.kernels.Linear()
prior = gpx.Prior(mean_function=mean, kernel=kernel)

x = jnp.linspace(-3.0, 3.0, num=200).reshape(-1, 1)
rv = prior(x)
y = rv.sample(seed=key, sample_shape=(10,))

fig, ax = plt.subplots()
ax.plot(x, y.T, alpha=0.7)
ax.set_title("Samples from the Linear Kernel")
plt.show()

# %% [markdown]
# ## What are the Necessary Conditions for a Valid Kernel?

# %% [markdown]
# In this guide we have introduced several different kernel functions, $k$, which may make you wonder if any function of two input pairs you construct will make a valid kernel function? Alas, not any function can be used as a kernel function in a GP, and there is a necessary condition a function must satisfy in order to be a valid kernel function.
#
# In order to understand the necessary condition, it is useful to introduce the idea of a *Gram matrix*. As introduced in the [GP introduction notebook](https://docs.jaxgaussianprocesses.com/examples/intro_to_gps/), given $n$ input points, $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$, the *Gram matrix* is defined as:
#
# $$K(\mathbf{X}, \mathbf{X}) = \begin{bmatrix} k(\mathbf{x}_1, \mathbf{x}_1) & \cdots & k(\mathbf{x}_1, \mathbf{x}_n) \\ \vdots & \ddots & \vdots \\ k(\mathbf{x}_n, \mathbf{x}_1) & \cdots & k(\mathbf{x}_n, \mathbf{x}_n) \end{bmatrix}$$
#
# such that $K(\mathbf{X}, \mathbf{X})_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$.
#
# In order for $k$ to be a valid kernel/covariance function, the corresponding covariance martrix must be *positive semi-definite*. A real $n \times n$ matrix $K$ is positive semi-definite if and only if for all vectors $\mathbf{z} \in \mathbb{R}^n$, $\mathbf{z}^\top K \mathbf{z} \geq 0$. Alternatively, a real $n \times n$ matrix $K$ is positive semi-definite if and only if all of its eigenvalues are non-negative.

# %% [markdown]
# ## Defining Kernels on Non-Euclidean Spaces
#
# In this notebook, we have focused solely on kernels whose domain resides in Euclidean space. However, what if one wished to work with data whose domain is non-Euclidean? For instance, one may wish to work with graph-structured data, or data which lies on a manifold, or even strings. Fortunately, kernels exist for a wide variety of domains. Whilst this is beyond the scope of this notebook, feel free to checkout out our [notebook on graph kernels](https://docs.jaxgaussianprocesses.com/examples/graph_kernels/) for an introduction on how to define the Matérn kernel on graph-structured data, and there are a wide variety of resources online for learning about defining kernels in other domains. In terms of open-source libraries, the [Geometric Kernels](https://github.com/GPflow/GeometricKernels) library could be a good place to start if you're interested in looking at how these kernels may be implemented, with the additional benefit that it is compatible with GPJax.

# %% [markdown]
# ## Further Reading
#
# Congratulations on making it this far! We hope that this guide has given you a good introduction to kernels and how they can be used in GPJax. If you're interested in learning more about kernels, we recommend the following resources, which have also been used as inspiration for this guide:
#
# - [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) - Chapter 4 provides a comprehensive overview of kernels, diving deep into some of the technical details and also providing some kernels defined on non-Euclidean spaces such as strings.
# - David Duvenaud's [Kernel Cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/) is a great resource for learning about kernels, and also provides some information about some of the pitfalls people commonly encounter when using the Matérn family of kernels. His PhD thesis, [Automatic Model Construction with Gaussian Processes](https://www.cs.toronto.edu/~duvenaud/thesis.pdf), also provides some in-depth recipes for how one may incorporate their prior knowledge when constructing kernels.
# - Finally, please check out our [more advanced kernel guide](https://docs.jaxgaussianprocesses.com/examples/kernels/), which details some more kernels available in GPJax as well as how one may combine kernels together to form more complex kernels.
#
# ## System Configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Christie'
