# %% [markdown]
# # Gaussian Processes for Vector Fields
# In this notebook, we use Gaussian processes to learn vector valued functions. We will be
# recreating the results by [Berlinghieri et. al, (2023)](https://arxiv.org/pdf/2302.10364.pdf).
# %%
from jax.config import config

config.update("jax_enable_x64", True)
import torch

torch.manual_seed(123)
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import install_import_hook
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as onp
import optax as ox
import tensorflow_probability as tfp
from jax.config import config
from dataclasses import dataclass
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
from jax import hessian, jit

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
key = jr.PRNGKey(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = rcParams["axes.prop_cycle"].by_key()["color"]

# jnp.set_printoptions(4, edgeitems=30, linewidth=100000,
#     formatter=dict(float=lambda x: "%.3g" % x))


# %% [markdown]
# ## Data Loading
# Data and original code are publicly available at https://github.com/renatoberlinghieri/Helmholtz-GP
#
# The data are 297 (artificial) measurements of ocean velocity, given by the vector field
#
# $$
# \mathbf{F}(\mathbf{x}) = -x^{(1)}\hat{\imath} + x^{(0)}\hat{\jmath},
# $$
#
#
#
# where $\mathbf{x} = (x^{(0)}$,$x^{(1)})^\text{T}$, $\hat{\imath}$ and $\hat{\jmath}$ are unit vectors in the standard Cartesian directions (dimensions will be indicated by superscripts).
#
# The measurements are contained in a dataset $D_0=\left\{ \left(\mathbf{x}_{0,i} , \mathbf{y}_{0,i} \right)\right\}_{i=1}^N$, where $\mathbf{y}_i$ is a 2 dimensional velocity vector and $\mathbf{x}_i$ is a 2 dimensional position vector on an $N=17\times17$ grid, equally spaced over the interval $[-1,1] \times[-1,1]$. 8 measurements of $\mathbf{F}(\mathbf{x})$ at new locations are contained in a dataset $D_T=\left\{\left(\mathbf{x}_{T,i}, \mathbf{y}_{T,i} \right)\right\}_{i=1}^8$, which are allocated to training data.
#
#

# %%
# loading in data
try:
    Pos_Train = onp.genfromtxt("data/XY_train_vortex.csv", delimiter=",").T
    Pos_Test = onp.genfromtxt("data/XY_test_vortex.csv", delimiter=",").T
    Vel_Train = onp.genfromtxt("data/UV_train_vortex.csv", delimiter=",").T
    Vel_Test = onp.genfromtxt("data/UV_test_vortex.csv", delimiter=",").T
except FileNotFoundError:
    Pos_Train = onp.genfromtxt(
        "docs/examples/data/XY_train_vortex.csv", delimiter=","
    ).T
    Pos_Test = onp.genfromtxt("docs/examples/data/XY_test_vortex.csv", delimiter=",").T
    Vel_Train = onp.genfromtxt(
        "docs/examples/data/UV_train_vortex.csv", delimiter=","
    ).T
    Vel_Test = onp.genfromtxt("docs/examples/data/UV_test_vortex.csv", delimiter=",").T


fig, ax = plt.subplots(1, 1)
ax.quiver(Pos_Test[0], Pos_Test[1], Vel_Test[0], Vel_Test[1], label="$D_0$")
ax.quiver(
    Pos_Train[0],
    Pos_Train[1],
    Vel_Train[0],
    Vel_Train[1],
    color="red",
    alpha=0.7,
    label="$D_T$",
)
ax.set(xlim=[-1.3, 1.3], ylim=[-1.3, 1.3], aspect="equal")
ax.legend()
plt.show()

# %% [markdown]
# ## Problem Setting
# Our aim is to obtain estimates for $\mathbf{F}$ at the set of points $\left\{ \mathbf{x}_{0,i} \right\}_{i=1}^N$ using Gaussian processes, followed by a comparison of the latent model to the ground truth ($D_0$). Note that the model only sees $D_T$, and $D_0$ is used in the final benchmark only.
#
# Since $\mathbf{F}$ is a vector-valued function, we would ideally require GPs that can learn vector-valued functions. Currently, GPJax only supports learning scalar-valued functions' GPs[<sup>1</sup>](#fn1) . However, since $D_T$ contains a 2D vector measurement $\mathbf{y}_{T,i}$ at each location $\mathbf{x}_{T,i}$, we may get around this by 'massaging' the data into a  $2N\times2N$ problem, such that each dimension of our GP is associated with a *component* of $\mathbf{y}_{T,i}$.
#
# For a particular  $\mathbf{y}$ (training or testing) at location $\mathbf{x}$, the components $(y^{(0)}, y^{(1)})$ are described by the latent vector field $\mathbf{F}$ such that
#
# $$
# \mathbf{y} = \mathbf{F}(\mathbf{x}) = \left(\begin{array}{l}
# f^{(0)}\left(\mathbf{x}\right) \\
# f^{(1)}\left(\mathbf{x}\right)
# \end{array}\right)
# $$
#
# where each $f^{(z)}\left(\mathbf{x}\right), z \in \{0,1\}$ is a scalar valued function. Now consider the scalar-valued function $g: \mathbb{R}^2 \times\{0,1\} \rightarrow \mathbb{R}$, such that
#
# $$
# g \left(\mathbf{x} , 0 \right) = f^{(0)} ( \mathbf{x} ), \text{and } g \left( \mathbf{x}, 1 \right)=f^{(1)}\left(\mathbf{x}\right).
# $$
#
# We have increased the input dimension by 1, from the 2D $\mathbf{x}$ to 3D $\left(\mathbf{x}, 0\right)$ or $\left(\mathbf{x}, 1\right)$
#
# By choosing the value of the third dimension, 0 or 1, we may now incorporate this information into computation of the kernel.
# We therefore make new 3D datasets $D_{T,3D} = \left\{\left( \mathbf{X}_{T,i},\mathbf{Y}_{T,i} \right) \right\} _{i=0}^{16}$ and $D_{0,3D} = \left\{\left( \mathbf{X}_{0,i},\mathbf{Y}_{0,i} \right) \right\} _{i=0}^{2N}$ that incorporates this new labelling, such that for each dataset (indicated by the subscript $D = 0$ or $D=T$),
#
# $$
# X_{D,i} = \left( \mathbf{x}_{D,i}, z \right),
# $$
# and
# $$
# Y_{D,i} = y_{D,i}^{(z)},
# $$
#
# where $z = 0$ if $i$ is odd and $z=1$ if $i$ is even.

# %%

# introduce alternating z label
NTrain = len(Pos_Train[0])
NTest = len(Pos_Test[0])
zLabelTrain = jnp.tile(jnp.array([0.0, 1.0]), NTrain)
zLabelTest = jnp.tile(jnp.array([0.0, 1.0]), NTest)
# Change vectors x -> X = (x,z), and vectors y -> Y = (y,z) using via the artificial z label
PosTrain3D = jnp.vstack((jnp.repeat(Pos_Train, repeats=2, axis=1), zLabelTrain)).T
VelTrain3D = Vel_Train.T.flatten().reshape(-1, 1)
# we also require the testing data to be relabelled for later use, such that we can query the 2Nx2N GP at the test points
PosTest3D = jnp.vstack(
    (jnp.repeat(Pos_Test.reshape(2, 289), repeats=2, axis=1), zLabelTest)
).T
# Pass the labelled training data into gpx.dataset object
D0 = gpx.Dataset(X=PosTrain3D, y=VelTrain3D)

# %% [markdown]
# ## Velocity (Dimension) Decomposition
# Having labelled the data, we are now in a position to use a GP to learn the function $g$, and hence $\mathbf{F}$. A naive approach to the problem is to apply a GP prior directly onto the velocities of each dimension independently, which is called the *velocity* GP. For our prior, we choose an isotropic mean 0 over all dimensions of the GP, and a piecewise kernel that depends on the $z$ labels of the inputs, such that for two inputs $\mathbf{X} = \left( \mathbf{x}, z \right )$ and $\mathbf{X}^\prime = \left( \mathbf{x}^\prime, z^\prime \right )$,
#
# $$
# k_{\text{vel}} \left(\mathbf{X}, \mathbf{X}^{\prime}\right)= \begin{cases}k^{(z)}\left(\mathbf{x}, \mathbf{x}^{\prime}\right) & \text { if } z=z^{\prime} \\ 0 & \text { if } z \neq z^{\prime}\end{cases}
# $$
#
# where $k^{(z)}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)$ are the user are kernels for each dimension. What this means is that there are no correlations between the $x^{(0)}$ and $x^{(1)}$ dimensions for all choices $\mathbf{X}$ and $\mathbf{X}^{\prime}$, since there are no off-diagonal elements in the Gram matrix populated by this choice.
#
# To implement this approach in GPJax, we define `VelocityKernel` in the following cell, following the steps outlined in the creating new kernels notebook. This modular implementation takes the choice of user kernels as its class attributes: `kernel0` and `kernel1`. We must additionally pass the argument `active_dims = [0,1]`, which is an attribute of the base class `AbstractKernel`, into the chosen kernels. This is necessary such that the subsequent likelihood optimisation does not optimise over the artificial label dimension.
#

# %%
from jax.numpy import exp


@dataclass
class VelocityKernel(gpx.kernels.AbstractKernel):
    kernel1: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1])
    kernel0: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1])

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0. acheive the correct value via 'Switches'

        z = X[2]
        zp = Xp[2]

        k0Switch = (z + 1) % 2 * (zp + 1) % 2
        k1Switch = z * zp

        return k0Switch * self.kernel0(X, Xp) + k1Switch * self.kernel1(X, Xp)


# %% [markdown]
# ### GPJax Implementation
# Next, we define the model in GPJax. The prior is defined using $k_{\text{vel}}\left(\mathbf{X}, \mathbf{X}^\prime \right)$ and 0 mean. We choose a Gaussian likelihood.
#

# %%
# Define the Helmholtz GP
kernel = VelocityKernel()
mean = gpx.mean_functions.Zero()
Prior = gpx.Prior(mean_function=mean, kernel=kernel)
Likelihood = gpx.Gaussian(
    num_datapoints=D0.n, obs_noise=jnp.array([1.0], dtype=jnp.float64)
)
Posterior = Prior * Likelihood


# %% [markdown]
# With a model now defined, we can proceed to optimise the hyperparameters of our likelihood over $D_0$. This is done by minimising the marginal log likelihood using `optax`. We also plot its value at each step to visually confirm that we have found the minimum

# %%

# define the Marginal Log likelihood using D0
Objective = gpx.objectives.ConjugateMLL(negative=True)
Objective(Posterior, train_data=D0)
Optimiser = ox.adam(learning_rate=0.01)
Objective = jit(Objective)

NIters = 10000
# Optimise to minimise the MLL
OptPosteriorVel, history = gpx.fit(
    model=Posterior,
    objective=Objective,
    train_data=D0,
    optim=Optimiser,
    num_iters=NIters,
    safe=True,
    key=key,
)

fig, ax = plt.subplots(1, 1)
ax.plot(history, color="red")
ax.set(xlabel="Training iteration", ylabel="Negative marginal log likelihood")
plt.show()

# %% [markdown]
# ### Comparison
# We next obtain the latent distribution of the GP of $g$ at $\mathbf{x}_{0,i}$, then extract its mean and standard at the test locations, $\mathbf{F}_{\text{latent}}(\mathbf{x}_{0,i})$, as well as the standard deviation (we will use it at the very end).

# %%
VelocityLatent = OptPosteriorVel.predict(PosTest3D, train_data=D0)
VelocityMean = VelocityLatent.mean()
VelocityStd = VelocityLatent.stddev()
# extract x0 and x1 values over g
Vel_Lat = [VelocityMean[::2].reshape(17, 17), VelocityMean[1::2].reshape(17, 17)]
Pos_Lat = Pos_Test

# %% [markdown]
# We now replot the ground truth (testing data) $D_0$, the predicted latent vector field $\mathbf{F}_{\text{latent}}(\mathbf{x_i})$, and a heatmap of the residuals at each location $R(\mathbf{x}_i) = \left|\left| \mathbf{y}_{0,i} - \mathbf{F}_{\text{latent}}(\mathbf{x}_i) \right|\right|$.

# %%

# Residuals between ground truth and estimate
ResidualsVel = jnp.sqrt(
    (Vel_Test[0].reshape(17, 17) - Vel_Lat[0]) ** 2
    + (Vel_Test[1].reshape(17, 17) - Vel_Lat[1]) ** 2
)
X = Pos_Test[0]
Y = Pos_Test[1]

# make figure
fig, ax = plt.subplots(1, 3)
fig.tight_layout()
# ground truth
ax[0].quiver(
    Pos_Train[0],
    Pos_Train[1],
    Vel_Train[0],
    Vel_Train[1],
    color="red",
    label="Testing Data",
)
ax[0].quiver(Pos_Test[0], Pos_Test[1], Vel_Test[0], Vel_Test[1], label="Training data")
ax[0].set(xlim=[-1.3, 1.3], ylim=[-1.3, 1.3], aspect="equal", title="Ground Truth")

# Latent estimate of vector field F
ax[1].quiver(
    Pos_Lat[0],
    Pos_Lat[1],
    Vel_Lat[0],
    Vel_Lat[1],
    color="darkblue",
    label="Latent estimate of $\mathbf{F}$",
)
ax[1].quiver(Pos_Train[0], Pos_Train[1], Vel_Train[0], Vel_Train[1], color="red")
ax[1].set(
    xlim=[-1.3, 1.3], ylim=[-1.3, 1.3], aspect="equal", title="GP Latent Estimate"
)

im = ax[2].imshow(ResidualsVel, extent=[X.min(), X.max(), Y.min(), Y.max()], cmap="hot")
ax[2].set(xlim=[-1.3, 1.3], ylim=[-1.3, 1.3], aspect="equal", title="Residuals")
fig.colorbar(im, fraction=0.046, pad=0.04, orientation="vertical")
plt.show()

# %% [markdown]
# From the residuals it is evident the GP does not perform well far from the training data. This is because our construction of the kernel placed an independent prior on each physical dimension. This is incorrect, as by definition $f^{(z)}$ is directly proportional to $x^{(z^\prime)}$ for both physical dimensions. Therefore, we need a different approach that can implicitly incorporate this at a fundamental level. To achieve this we will require a *Helmholtz Decomposition*.

# %% [markdown]
# ## Helmholtz Decomposition
# In 2 dimensions, a twice continuously differentiable and compactly supported vector field $\mathbf{F}: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ can be expressed as the sum of the gradient of a scalar potential $\Phi: \mathbb{R}^2 \rightarrow \mathbb{R}$, called the potential function, and the vorticity operator of another scalar potential $\Psi: \mathbb{R}^2 \rightarrow \mathbb{R}$, called the stream function ([Berlinghieri et. al, (2023)](https://arxiv.org/pdf/2302.10364.pdf)):
# $$
# \mathbf{F}=\operatorname{grad} \Phi+\operatorname{rot} \Psi
# $$
# where
# $$
# \operatorname{grad} \Phi:=\left[\begin{array}{l}
# \partial \Phi / \partial x^{(0)} \\
# \partial \Phi / \partial x^{(1)}
# \end{array}\right] \text { and } \operatorname{rot} \Psi:=\left[\begin{array}{c}
# \partial \Psi / \partial x^{(1)} \\
# -\partial \Psi / \partial x^{(0)}
# \end{array}\right]
# $$
#
# This is reminiscent of a 3 dimensional [Helmholtz decomposition](https://en.wikipedia.org/wiki/Helmholtz_decomposition).
#
# The 2 dimensional decomposition motivates a different approach: placing priors on $\Psi$ and $\Phi$, allowing us to make assumptions directly about fundamental properties of $\mathbf{F}$. If we choose independent GP priors such that $\Phi \sim \mathcal{G P}\left(0, k_{\Phi}\right)$ and $\Psi \sim \mathcal{G P}\left(0, k_{\Psi}\right)$, then $\mathbf{F} \sim \mathcal{G P} \left(0, k_\text{Helm}\right)$ (since acting linear operations on a GPs give GPs).
#
# For $\mathbf{X}, \mathbf{X}^{\prime} \in \mathbb{R}^2 \times \left\{0,1\right\}$ and $z, z^\prime \in \{0,1\}$,
#
# $$
# \boxed{ k_{\mathrm{Helm}}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)_{z,z^\prime} =  \frac{\partial^2 k_{\Phi}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)}{\partial x^{(z)} \partial\left(x^{\prime}\right)^{(z^\prime)}}+(-1)^{z+z^\prime} \frac{\partial^2 k_{\Psi}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)}{\partial x^{(1-z)} \partial\left(x^{\prime}\right)^{(1-z^\prime)}}}.
# $$
#
# where $x^{(z)}$ and $(x^\prime)^{(z^\prime)}$ are the $z$ and $z^\prime$ components of $\mathbf{X}$ and ${\mathbf{X}}^{\prime}$ respectively.
#
# We compute the second derivatives using `jax.hessian`. In the following implementation, for a kernel $k(\mathbf{x}, \mathbf{x}^{\prime})$, this computes the Hessian matrix with respect to the components of $\mathbf{x}$
#
# $$
# \frac{\partial^2 k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)}{\partial x^{(z)} \partial x^{(z^\prime)}}.
# $$
#
# Note that we have operated $\dfrac{\partial}{\partial x^{(z)}}$, *not* $\dfrac{\partial}{\partial \left(x^\prime \right)^{(z)}}$, as the boxed equation suggests. This is not an issue if we choose stationary kernels $k(\mathbf{x}, \mathbf{x}^{\prime}) = k(\mathbf{x} - \mathbf{x}^{\prime})$ , as the partial derivatives with respect to the components have the following exchange symmetry:
#
# $$
# \frac{\partial}{\partial x^{(z)}} = - \frac{\partial}{\partial \left( x^\prime \right)^{(z)}}
# $$
#
# for either $z$. We specify that this implementation only works with stationary kernels by setting the type of `kernelPhi` and `kernelPsi` to `gpx.kernels.stationary`.
# %%


@dataclass
class HelmholtzKernel(gpx.kernels.AbstractKernel):
    # initialise Phi and Psi kernels as any stationary kernel in gpJax
    kernelPhi: gpx.kernels.stationary = gpx.kernels.RBF(active_dims=[0, 1])
    kernelPsi: gpx.kernels.stationary = gpx.kernels.RBF(active_dims=[0, 1])

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # obtain indices for k_helm, implement in the correct sign between the derivatives
        z = jnp.array(X[2], dtype=int)
        zp = jnp.array(Xp[2], dtype=int)
        Sign = (-1) ** (z + zp)
        # convert to array to correctly index, -ve sign due to exchange symmetry (only true for stationary kernels)
        DivPhi = -jnp.array(hessian(self.kernelPhi)(X, Xp), dtype=jnp.float64)[z][zp]
        DivPsi = -jnp.array(hessian(self.kernelPsi)(X, Xp), dtype=jnp.float64)[1 - z][
            1 - zp
        ]

        return DivPhi + Sign * DivPsi


# %% [markdown]
# ### GPJax Implementation
# We repeat the exact same steps as with the velocity GP model, but replacing `VelocityKernel` with `HelmholtzKernel`.

# %%
# Redefine Gaussian process with Helmholtz kernel
kernel = HelmholtzKernel()
Prior = gpx.Prior(mean_function=mean, kernel=kernel)
Likelihood = gpx.Gaussian(
    num_datapoints=D0.n, obs_noise=jnp.array([1.0], dtype=jnp.float64)
)
Posterior = Prior * Likelihood

# Optimise hyperparameters using optax
Objective = gpx.objectives.ConjugateMLL(negative=True)
Objective(Posterior, train_data=D0)
Optimiser = ox.adam(learning_rate=0.01)
Objective = jit(Objective)

OptPosteriorHelm, history = gpx.fit(
    model=Posterior,
    objective=Objective,
    train_data=D0,
    optim=Optimiser,
    num_iters=NIters,
    safe=True,
    key=key,
)

fig, ax = plt.subplots(1, 1)
ax.plot(history, color="red")
ax.set(xlabel="Training iteration", ylabel="Negative marginal log likelihood")
plt.show()

# extract the mean
HelmholtzLatent = OptPosteriorHelm.predict(PosTest3D, train_data=D0)
HelmholtzMean = HelmholtzLatent.mean()
HelmholtzStd = HelmholtzLatent.stddev()
# extract x and y values over g
Helm_Lat = [HelmholtzMean[::2].reshape(17, 17), HelmholtzMean[1::2].reshape(17, 17)]
Pos_Lat = Pos_Test

# %% [markdown]
# ### Comparison
# We again plot the ground truth (testing data) $D_0$, the predicted latent vector field $\mathbf{F}_{\text{latent}}(\mathbf{x_i})$, and a heatmap of the residuals at each location $R(\mathbf{x}_i) = \left|\left| \mathbf{y}_{0,i} - \mathbf{F}_{\text{latent}}(\mathbf{x}_i) \right|\right|$.

# %%

# Residuals between ground truth and estimate
ResidualsHelm = jnp.sqrt(
    (Vel_Test[0].reshape(17, 17) - Helm_Lat[0]) ** 2
    + (Vel_Test[1].reshape(17, 17) - Helm_Lat[1]) ** 2
)
X = Pos_Test[0]
Y = Pos_Test[1]

# make figure
fig, ax = plt.subplots(1, 3)
fig.tight_layout()
# ground truth
ax[0].quiver(
    Pos_Train[0],
    Pos_Train[1],
    Vel_Train[0],
    Vel_Train[1],
    color="red",
    label="Testing Data",
)
ax[0].quiver(Pos_Test[0], Pos_Test[1], Vel_Test[0], Vel_Test[1], label="Training data")
ax[0].set(xlim=[-1.3, 1.3], ylim=[-1.3, 1.3], aspect="equal", title="Ground Truth")
# Latent estimate of vector field F
ax[1].quiver(
    Pos_Lat[0],
    Pos_Lat[1],
    Helm_Lat[0],
    Helm_Lat[1],
    color="darkblue",
    label="Latent estimate of $\mathbf{F}$",
)
ax[1].quiver(Pos_Train[0], Pos_Train[1], Vel_Train[0], Vel_Train[1], color="red")
ax[1].set(
    xlim=[-1.3, 1.3],
    ylim=[-1.3, 1.3],
    aspect="equal",
    title="GP Latent Estimate of $\mathbf{F}$",
)
# Residuals
im = ax[2].imshow(
    ResidualsHelm, extent=[X.min(), X.max(), Y.min(), Y.max()], cmap="hot"
)
ax[2].set(xlim=[-1.3, 1.3], ylim=[-1.3, 1.3], aspect="equal", title="Residuals")
fig.colorbar(im, fraction=0.046, pad=0.04, orientation="vertical")
plt.show()

# %% [markdown]
# Visually, the Helmholtz model performs better than the velocity model, preserving the local structure of the $\mathbf{F}$, supportd by the residuals being much smaller than with the velocity decomposition. Since we placed priors on $\Phi$ and $\Psi$, the construction of $\mathbf{F}$ allows for correlations between the dimensions (non-zero off diagonal elements in the Gram matrix populated by $k_\text{Helm}\left(\mathbf{X},\mathbf{X}^{\prime}\right)$ ).


# %% [markdown]
# ## Negative Log Predictive Densities
# Lastly, we directly compare the velocity and Hemlholtz models by computing the [negative log predictive densities](https://en.wikipedia.org/wiki/Negative_log_predictive_density) for each model. This is a quantitative metric that measures the probability of the ground truth given the data,
#
# $$
# \mathrm{NLPD}=-\sum_{i=1}^{2N} \log \left(  p\left(\mathcal{Y}_i = Y_{0,i} \mid \mathbf{X}_{i}\right) \right)
# $$
#
# where each $p\left(\mathcal{Y}_i \mid \mathbf{X}_i \right)$ is the marginal Gaussian distribution at each test location, and $Y_{i,0}$ is the $i$th component of the (massaged) test data that we reserved at the beginning of the notebook in $D_0$. A smaller value is better, since the deviation of the ground truth and the model are small in this case.

# %%
# ensure testing data alternates between x0 and x1 components
Vel_Query = jnp.column_stack((Vel_Test[0], Vel_Test[1])).flatten()

NormalVel = tfp.substrates.jax.distributions.Normal(
    loc=VelocityMean,
    scale=VelocityStd,
)
NormalHelm = tfp.substrates.jax.distributions.Normal(
    loc=HelmholtzMean,
    scale=HelmholtzStd,
)

NLPDVel = -jnp.sum(NormalVel.log_prob(Vel_Query))
NLPDHelm = -jnp.sum(NormalHelm.log_prob(Vel_Query))

print("NLPD for Velocity: %.2f \nNLPD for Helmholtz: %.2f" % (NLPDVel, NLPDHelm))
# %% [markdown]
# The Helmholtz model significantly outperforms the velocity model, as indicated by the lower NLPD score.


# %% [markdown]
# <span id="fn1"></span>
# ## Footnote
# Kernels for vector valued functions have been studied in the literature, see [Alvarez et. al, (2012)](https://doi.org/10.48550/arXiv.1106.6251)
# ## System Configuration
# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Ivan Shalashilin'
