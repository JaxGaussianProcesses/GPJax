# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax as ox
from jax.config import config
from jaxutils import Dataset
import jaxkern as jk

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
# %% [markdown]
# # UCI Data Benchmarking
#
# In this notebook, we will show how to apply GPJax on a benchmark UCI regression problem. These kind of tasks are often used in the research community to benchmark and assess new techniques against those already in the literature. Much of the code contained in this notebook can be adapted to applied problems concerning datasets other than the one presented here.
# %%
import pandas as pd
from jax import jit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import gpjax as gpx

key = jr.PRNGKey(123)

# %% [markdown]
# ## Data Loading
#
# We'll be using the [Yacht](https://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics) dataset from the UCI machine learning data repository. Each observation describes the hydrodynamic performance of a yacht through its resistance. The dataset contains 6 covariates and a single positive, real valued response variable. There are 308 observations in the dataset, so we can comfortably use a conjugate regression Gaussian process here (for more more details, checkout the [Regression notebook](https://gpjax.readthedocs.io/en/latest/nbs/regression.html)).

# %%
yacht = pd.read_fwf("data/yacht_hydrodynamics.data", header=None).values[:-1, :]
X = yacht[:, :-1]
y = yacht[:, -1].reshape(-1, 1)

# %% [markdown]
# ## Preprocessing
#
# With a dataset loaded, we'll now preprocess it such that it is more amenable to modelling with a Gaussian process.
#
# ### Data Partitioning
#
# We'll first partition our data into a _training_ and _testing_ split. We'll fit our Gaussian process to the training data and evaluate its performance on the test data. This allows us to investigate how effectively our Gaussian process generalises to out-of-sample datapoints and ensure that we are not overfitting. We'll hold 30% of our data back for testing purposes.

# %%
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# ### Response Variable
#
# We'll now process our response variable $\mathbf{y}$. As the below plots show, the data has a very long tail and is certainly not Gaussian. However, we would like to model a Gaussian response variable so that we can adopt a Gaussian likelihood function and leverage the model's conjugacy. To achieve this, we'll first log-scale the data, to bring the long right tail in closer to the data's mean. We'll then standardise the data such that is distributed according to a unit normal distribution. Both of these transformations are invertible through the log-normal expectation and variance formulae and the the inverse standardisation identity, should we ever need our model's predictions to be back on the scale of the original dataset.
#
# For transforming both the input and response variable, all transformations will be done with respect to the training data where relevant.

# %%
log_ytr = np.log(ytr)
log_yte = np.log(yte)

y_scaler = StandardScaler().fit(log_ytr)
scaled_ytr = y_scaler.transform(log_ytr)
scaled_yte = y_scaler.transform(log_yte)

# %% [markdown]
# We can see the effect of these transformations in the below three panels.

# %%
fig, ax = plt.subplots(ncols=3, figsize=(16, 4), tight_layout=True)
ax[0].hist(ytr, bins=30)
ax[0].set_title("y")
ax[1].hist(log_ytr, bins=30)
ax[1].set_title("log(y)")
ax[2].hist(scaled_ytr, bins=30)
ax[2].set_title("scaled log(y)")

# %% [markdown]
# ### Input Variable
#
# We'll now transform our input variable $\mathbf{X}$ to be distributed according to a unit Gaussian.

# %%
x_scaler = StandardScaler().fit(Xtr)
scaled_Xtr = x_scaler.transform(Xtr)
scaled_Xte = x_scaler.transform(Xte)

# %% [markdown]
# ## Model fitting
#
# With data now loaded and preprocessed, we'll proceed to defining a Gaussian process model and optimising its parameters. This notebook purposefully does not go into great detail on this process, so please see notebooks such as the [Regression notebook](https://gpjax.readthedocs.io/en/latest/nbs/regression.html) and [Classification notebook](https://gpjax.readthedocs.io/en/latest/nbs/classification.html) for further information.
#
# ### Model specification
#
# We'll use a radial basis function kernel to parameterise the Gaussian process in this notebook. As we have 5 covariates, we'll assign each covariate its own lengthscale parameter. This form of kernel is commonly known as an automatic relevance determination (ARD) kernel.
#
# In practice, the exact form of kernel used should be selected such that it represents your understanding of the data. For example, if you were to model temperature; a process that we know to be periodic, then you would likely wish to select a periodic kernel. Having _Gaussian-ised_ our data somewhat, we'll also adopt a Gaussian likelihood function.

# %%
n_train, n_covariates = scaled_Xtr.shape
kernel = jk.kernels.RBF(active_dims=list(range(n_covariates)))
prior = gpx.Prior(kernel=kernel)

likelihood = gpx.Gaussian(num_datapoints=n_train)

posterior = prior * likelihood

# %% [markdown]
# ### Model Optimisation
#
# With a model now defined, we can proceed to optimise the hyperparameters of our model using Optax.

# %%
training_data = Dataset(X=scaled_Xtr, y=scaled_ytr)

parameter_state = gpx.initialise(posterior, key)
negative_mll = jit(
    posterior.marginal_log_likelihood(train_data=training_data, negative=True)
)
optimiser = ox.adam(0.05)

inference_state = gpx.fit(
    objective=negative_mll,
    parameter_state=parameter_state,
    optax_optim=optimiser,
    num_iters=1000,
    log_rate=50,
)

learned_params, training_history = inference_state.unpack()

# %% [markdown]
# ## Prediction
#
# With an optimal set of parameters learned, we can make predictions on the set of data that we held back right at the start. We'll do this in the usual way by first computing the latent function's distribution before computing the predictive posterior distribution.

# %%
latent_dist = posterior(
    learned_params,
    training_data,
)(scaled_Xte)
predictive_dist = likelihood(learned_params, latent_dist)

predictive_mean = predictive_dist.mean()
predictive_stddev = predictive_dist.stddev()

# %% [markdown]
# ## Evaluation
#
# We'll now show how the performance of our Gaussian process can be evaluated by numerically and visually.
#
# ### Metrics
#
# To numerically assess the performance of our model, two commonly used metrics are root mean squared error (RMSE) and the R2 coefficient. RMSE is simply the square root of the squared difference between predictions and actuals. A value of 0 for this metric implies that our model has 0 generalisation error on the test set. R2 measures the amount of variation within the data that is explained by the model. This can be useful when designing variance reduction methods such as control variates as it allows you to understand what proportion of the data's variance will be soaked up. A perfect model here would score 1 for R2 score, whereas predicting the data's mean would score 0 and models doing worse than simple mean predictions can score less than 0.

# %%
rmse = mean_squared_error(y_true=scaled_yte.squeeze(), y_pred=predictive_mean)
r2 = r2_score(y_true=scaled_yte.squeeze(), y_pred=predictive_mean)
print(f"Results:\n\tRMSE: {rmse: .4f}\n\tR2: {r2: .2f}")

# %% [markdown]
# Both of these metrics seem very promising, so, based off these, we can be quite happy that our first attempt at modelling the Yacht data is promising.
#
# ### Diagnostic plots
#
# To accompany the above metrics, we can also produce residual plots to explore exactly where our model's shortcomings lie. If we define a residual as the true value minus the prediction, then we can produce three plots:
#
# 1. Predictions vs. actuals.
# 2. Predictions vs. residuals.
# 3. Residual density.
#
# The first plot allows us to explore if our model struggles to predict well for larger or smaller values by observing where the model deviates more from the line $y=x$. In the second plot we can inspect whether or not there were outliers or structure within the errors of our model. A well-performing model would have predictions close to and symmetrically distributed either side of $y=0$. Such a plot can be useful for diagnosing heteroscedasticity. Finally, by plotting a histogram of our residuals we can observe whether or not there is any skew to our residuals.

# %%
residuals = scaled_yte.squeeze() - predictive_mean

fig, ax = plt.subplots(ncols=3, figsize=(16, 4), tight_layout=True)

ax[0].scatter(predictive_mean, scaled_yte.squeeze())
ax[0].plot([0, 1], [0, 1], color="tab:orange", transform=ax[0].transAxes)
ax[0].set(xlabel="Predicted", ylabel="Actual", title="Predicted vs Actual")

ax[1].scatter(predictive_mean.squeeze(), residuals)
ax[1].plot([0, 1], [0.5, 0.5], color="tab:orange", transform=ax[1].transAxes)
ax[1].set_ylim([-1.0, 1.0])
ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Predicted vs Residuals")

ax[2].hist(np.asarray(residuals), bins=30)
ax[2].set_title("Residuals")

# %% [markdown]
# From this, we can see that our model is struggling to predict the smallest values of the Yacht's hydrodynamic and performs increasingly well as the Yacht's hydrodynamic performance increases. This is likely due to the original data's heavy right-skew, and successive modelling attempts may wish to introduce a heteroscedastic likelihood function that would enable more flexible modelling of the smaller response values.
#
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
