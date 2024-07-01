import jax
import jax.numpy as jnp
import jax.random as jr


#with install_import_hook("gpjax", "beartype.beartype"):
import gpjax as gpx
import matplotlib.pyplot as plt

key = jr.PRNGKey(123)

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions



# custom bits
from utils import VerticalDataset, ProblemInfo
from models import ConjugatePrecipGP






def plot_params(problem_info:ProblemInfo, model,data,title=""):
    if isinstance(model,ConjugatePrecipGP):
        plt.figure()
        if jnp.shape(model.kernel.lengthscale)[0]>1:
            lengthscales = model.kernel.lengthscale
        else:
            lengthscales = jnp.array([model.kernel.lengthscale[0]]*data.dim)
            
        z_to_plot = jnp.linspace(jnp.min(model.smoother.Z_levels),jnp.max(model.smoother.Z_levels),100)
        smoothing_weights = model.smoother.smooth_fn(z_to_plot) 
        z_unscaled = z_to_plot * problem_info.pressure_std+ problem_info.pressure_mean
        for i in range(problem_info.num_3d_variables):
            plt.plot(smoothing_weights[i,:].T,z_unscaled, label=f"{problem_info.names_3d_short[i]} with lengthscales_ {lengthscales[i]:.2f}")
        plt.legend()
        plt.title(title+f" other lengthscales are {lengthscales[problem_info.num_3d_variables:]}")







def plot_data(problem_info:ProblemInfo, data:VerticalDataset):
    plt.hist(data.y.T)
    plt.title("Y")

    for X in [data.X3d]:
        fig, ax = plt.subplots(nrows=3, ncols=4)
        i,j=0,0
        for row in ax:
            for col in row:
                col.boxplot(X[:,i,:].T, showfliers=False);
                col.set_title(problem_info.names_3d_short[i])
                i+=1
                if i==X.shape[1]:
                    break
            if i==X.shape[1]:
                break

    fig, ax = plt.subplots(nrows=1, ncols=max(data.X2d.shape[1],2))
    for i in range(data.X2d.shape[1]):
        ax[i].hist(data.X2d[:100000,i].T);
        ax[i].set_title(problem_info.names_2d_short[i])
        #ax[i].set_xlim(0.0,1.0)
    fig, ax = plt.subplots(nrows=1, ncols=max(data.Xstatic.shape[1],2))
    i=0
    for i in range(data.Xstatic.shape[1]):
        ax[i].hist(data.Xstatic[:100000,i].T);
        ax[i].set_title(problem_info.names_static_short[i])
        #ax[i].set_xlim(0.0,1.0)

