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
from gpjax.precip_gp import VerticalDataset, ProblemInfo







def plot_params(problem_info:ProblemInfo, model,data,title="", print_corr=False):
    if isinstance(model, gpx.variational_families.AbstractVariationalFamily):
        model = model.posterior
    plt.figure()
    lengthscales = jnp.array([model.base_kernels[i].lengthscale[0] for i in range(data.dim)])
    z_to_plot = jnp.linspace(jnp.min(model.smoother.Z_levels),jnp.max(model.smoother.Z_levels),100)
    smoothing_weights = model.smoother.smooth_fn(z_to_plot) 
    z_unscaled = z_to_plot * problem_info.pressure_std+ problem_info.pressure_mean
    for i in range(problem_info.num_3d_variables):
        plt.plot(smoothing_weights[i,:].T,z_unscaled, label=f"{problem_info.names_3d_short[i]} with lengthscales_ {lengthscales[i]:.2f}")
    plt.legend()
    plt.title(title+f" other lengthscales are {lengthscales[problem_info.num_3d_variables:]}")
    smoothed = model.smoother.smooth_data(data)[0]
    plt.figure()
    for i in range(problem_info.num_3d_variables):
        plt.hist(smoothed[i,:], label=problem_info.names_3d_short[i], alpha=0.5)
    plt.legend()


    corr = jnp.corrcoef(smoothed.T)
    plt.figure()
    plt.imshow(corr)
    plt.colorbar()
    plt.figure()
    # plt.hist(corr.flatten(), bins=10);
    pairs = []
    for i in range(corr.shape[0]):
        for j in range(i):
            if jnp.absolute(corr[i][j])>0.75:
                pairs.append([problem_info.names[i], problem_info.names[j]])
                if print_corr:
                    print(f"{problem_info.names[i]} and {problem_info.names[j]} have correlation {corr[i][j]}")



def plot_interactions(problem_info:ProblemInfo, model, data, k=10,use_range=False):
        
    plt.figure()
    idx_2 = []
    for i in range(problem_info.num_variables):
        for j in range(i+1,problem_info.num_variables):
            idx_2.append([i,j])
    idxs = [[]] + [[i] for i in range(problem_info.num_variables)] + idx_2
    if isinstance(model, gpx.variational_families.AbstractVariationalFamily):
        sobols = model.get_sobol_indicies(idxs,use_range=use_range)
        z = model.inducing_inputs
    else:
        sobols = model.get_sobol_indicies(data, idxs,use_range=use_range)
        z = model.smoother.smooth_data(data)[0]
    sobols = sobols / jnp.sum(sobols)

    plt.plot(sobols)
    plt.title("sobol indicies (red lines between orders)")
    plt.axvline(x=1, color="red")
    plt.axvline(x=problem_info.num_variables+1, color="red")
    for idx in jax.lax.top_k(sobols, k)[1]:
        chosen_idx = idxs[idx]
        plt.figure()
        num_plot = 1_000 if len(chosen_idx)==1 else 10_000
        from scipy.stats import qmc
        sampler = qmc.Halton(d=problem_info.num_variables)
        x_plot = sampler.random(n=num_plot) * (jnp.max(z, axis=0) - jnp.min(z, axis=0)) + jnp.min(z, axis=0)
        if len(chosen_idx)==1:
            if isinstance(model, gpx.variational_families.AbstractVariationalFamily):
                mean = model.predict_indiv_mean(x_plot,chosen_idx)
                std = jnp.sqrt(model.predict_indiv_var(x_plot,chosen_idx))
            else:
                mean = model.predict_indiv_mean(x_plot, data, chosen_idx)
                std = jnp.sqrt(model.predict_indiv_var(x_plot, data, chosen_idx))
            mean = mean * (data.Y_std)#+ data.Y_mean
            std = std* (data.Y_std)
            plt.scatter(x_plot[:,chosen_idx[0]],mean, color="blue") 
            plt.scatter(x_plot[:,chosen_idx[0]],mean+ 1.96*std, color="red") 
            plt.scatter(x_plot[:,chosen_idx[0]],mean- 1.96*std, color="red") 
            plt.xlim([jnp.min(x_plot[:,chosen_idx[0]]),jnp.max(x_plot[:,chosen_idx[0]])])
            plt.scatter(z[:,chosen_idx[0]],jnp.zeros_like(z[:,chosen_idx[0]]), color="black")
            plt.title(f"Best guess (and uncertainty) at additive contributions from {[problem_info.names[i] for i in chosen_idx]}with sobol index {sobols[idx]}")
        elif len(chosen_idx)==2:
            if isinstance(model, gpx.variational_families.AbstractVariationalFamily):
                mean = model.predict_indiv_mean(x_plot,chosen_idx)
            else:
                mean = model.predict_indiv_mean(x_plot, data, chosen_idx)
            mean = mean * (data.Y_std)# + data.Y_mean
            col = plt.scatter(x_plot[:,chosen_idx[0]],x_plot[:,chosen_idx[1]],c=mean)
            plt.ylim([jnp.min(z[:,chosen_idx[1]]),jnp.max(z[:,chosen_idx[1]])])
            plt.colorbar(col)
            plt.scatter(z[:,chosen_idx[0]],z[:,chosen_idx[1]], color="black")
            plt.title(f"Best guess at additive contribution from {[problem_info.names[i] for i in chosen_idx]} with sobol index {sobols[idx]}")
        plt.xlim([jnp.min(x_plot[:,chosen_idx[0]]),jnp.max(x_plot[:,chosen_idx[0]])])   
       
        
        
def plot_component(problem_info:ProblemInfo, model, data,chosen_idx):
        plt.figure()
        
        if isinstance(model, gpx.variational_families.AbstractVariationalFamily):
            z = model.inducing_inputs
        else:
            z = model.smoother.smooth_data(data)[0]
        
        num_plot = 1_000 if len(chosen_idx)==1 else 10_000
        from scipy.stats import qmc
        sampler = qmc.Halton(d=problem_info.num_variables)
        x_plot = sampler.random(n=num_plot) * (jnp.max(z, axis=0) - jnp.min(z, axis=0)) + jnp.min(z, axis=0)

        if len(chosen_idx) in [0,1]:
            if isinstance(model, gpx.variational_families.AbstractVariationalFamily):
                mean = model.predict_indiv_mean(x_plot,chosen_idx)
                std = jnp.sqrt(model.predict_indiv_var(x_plot,chosen_idx))
            else:
                mean = model.predict_indiv_mean(x_plot, data, chosen_idx)
                std = jnp.sqrt(model.predict_indiv_var(x_plot, data, chosen_idx))
            mean = mean * (data.Y_std)#+ data.Y_mean
            std = std* (data.Y_std)
            if len(chosen_idx)==0:
                plt.scatter(x_plot[:,0],mean, color="blue")
                plt.scatter(x_plot[:,0
                                   ],mean+ 1.96*std, color="red")
                plt.scatter(x_plot[:,0],mean- 1.96*std, color="red")
            else:
                plt.scatter(x_plot[:,chosen_idx[0]],mean, color="blue") 
                plt.scatter(x_plot[:,chosen_idx[0]],mean+ 1.96*std, color="red") 
                plt.scatter(x_plot[:,chosen_idx[0]],mean- 1.96*std, color="red") 
                plt.xlim([jnp.min(x_plot[:,chosen_idx[0]]),jnp.max(x_plot[:,chosen_idx[0]])])
                plt.scatter(z[:,chosen_idx[0]],jnp.zeros_like(z[:,chosen_idx[0]]), color="black")
                plt.title(f"Best guess (and uncertainty) at additive contributions from {[problem_info.names[i] for i in chosen_idx]}")
                plt.xlim([jnp.min(x_plot[:,chosen_idx[0]]),jnp.max(x_plot[:,chosen_idx[0]])])  
        elif len(chosen_idx)==2:
            if isinstance(model, gpx.variational_families.AbstractVariationalFamily):
                mean = model.predict_indiv_mean(x_plot,chosen_idx)
            else:
                mean = model.predict_indiv_mean(x_plot, data, chosen_idx)
            mean = mean * (data.Y_std)# + data.Y_mean
            col = plt.scatter(x_plot[:,chosen_idx[0]],x_plot[:,chosen_idx[1]],c=mean)
            plt.ylim([jnp.min(z[:,chosen_idx[1]]),jnp.max(z[:,chosen_idx[1]])])
            plt.colorbar(col)
            plt.scatter(z[:,chosen_idx[0]],z[:,chosen_idx[1]], color="black")
            plt.title(f"Best guess at additive contribution from {[problem_info.names[i] for i in chosen_idx]}")
            plt.xlim([jnp.min(x_plot[:,chosen_idx[0]]),jnp.max(x_plot[:,chosen_idx[0]])])   
       
        




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



def plot_smoothed(ax,data, title="",scale=1000):
        idx = jnp.argsort(data)
        data = data[idx]
        y = Y[idx,0]
        from scipy.ndimage import gaussian_filter1d
        y_smooth = gaussian_filter1d(y, scale)

        ax.scatter(data,y,alpha=0.1)
        ax2 = ax.twinx()
        ax2.scatter(data,y_smooth,color="red")
        ax.set_title(title)

def plot_marginals(problem_info:ProblemInfo, D,scale=1000):
    for i in range(problem_info.num_3d_variables):
        fig, ax= plt.subplots()
        for j in range(len(problem_info.pressure_levels[0])):
            plot_smoothed(ax,D.X3d[:,i,j], problem_info.names_3d_short[i], scale)
    for j in range(problem_info.num_3d_variables, problem_info.num_3d_variables+problem_info.num_2d_variables):
        fig, ax= plt.subplots()
        plot_smoothed(ax,D.X2d[:,j-problem_info.num_3d_variables], problem_info.names_2d_short[j-problem_info.num_3d_variables], scale)
    for k in range(problem_info.num_3d_variables+problem_info.num_2d_variables, problem_info.num_variables):
        fig, ax= plt.subplots()
        plot_smoothed(ax,D.Xstatic[:,k-problem_info.num_3d_variables-problem_info.num_2d_variables], problem_info.names_static_short[k-problem_info.num_3d_variables-problem_info.num_2d_variables], scale)

    


