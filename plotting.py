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
from utils import  ProblemInfo




def plot_data(problem_info:ProblemInfo, data:gpx.Dataset):
    plt.hist(data.y.T)
    plt.title("Y")


    fig, ax = plt.subplots(nrows=data.X.shape[1], ncols=1)
    for i in range(data.X.shape[1]):
        ax[i].hist(data.X[:100000,i].T);
        ax[i].set_title(problem_info.names_short[i])
        #ax[i].set_xlim(0.0,1.0)




def plot_interactions(problem_info:ProblemInfo, model, data, k=10,use_range=False, greedy=False, lim=None):
        
    
    idx_2 = []
    for i in range(problem_info.num_variables):
        for j in range(i+1,problem_info.num_variables):
            idx_2.append([i,j])
    idxs = [[i] for i in range(problem_info.num_variables)] 
    if model.max_interaction_depth==2:
        idxs = idxs + idx_2

    sobols_for_plot = model.get_sobol_indicies(data, idxs, use_range=use_range, greedy=False)
    sobols_for_plot = sobols_for_plot / jnp.sum(sobols_for_plot,-1, keepdims=True) # [2,c]
    if greedy:
        sobols = model.get_sobol_indicies(data, idxs, use_range=use_range, greedy=True)   
    else:
        sobols = sobols_for_plot

    z = data.X
    zmax = jnp.max(z, axis=0)
    zmin = jnp.min(z, axis=0)


    plt.figure()
    plt.plot(sobols_for_plot)
    plt.title(f" sobol indicies (red lines between orders)")
    plt.axvline(x=0, color="red")
    plt.axvline(x=problem_info.num_variables, color="red")
    for idx in jax.lax.top_k(sobols, k)[1]:
        chosen_idx = idxs[idx]
        plt.figure()
        num_plot = 1_000 if len(chosen_idx)==1 else 5_000
        from scipy.stats import qmc
        sampler = qmc.Halton(d=problem_info.num_variables)
        x_plot = sampler.random(n=num_plot)* (zmax - zmin) + zmin
        if len(chosen_idx)==1:     
            mean= model.predict_indiv_mean(x_plot,data, chosen_idx)
            var = model.predict_indiv_var(x_plot,data, chosen_idx)
            std = jnp.sqrt(var)
            # mean = mean[j:j+1,:] #* (data.Y_std)#+ data.Y_mean
            # std = std[j:j+1,:]#* (data.Y_std)
            plt.scatter(x_plot[:,chosen_idx[0]],mean, color="blue") 
            plt.scatter(x_plot[:,chosen_idx[0]],mean+ 1.96*std, color="red") 
            plt.scatter(x_plot[:,chosen_idx[0]],mean- 1.96*std, color="red") 
            plt.scatter(z[:,chosen_idx[0]],jnp.zeros_like(z[:,chosen_idx[0]]), color="black",alpha=0.1)
            # ip = model.get_inducing_locations()
            # plt.xlim([jnp.min(jnp.hstack([x_plot[:,chosen_idx[0]],ip[:,chosen_idx[0]]])),jnp.max(jnp.hstack([x_plot[:,chosen_idx[0]],ip[:,chosen_idx[0]]]))])
            # plt.scatter(ip[:,chosen_idx[0]],jnp.zeros_like(ip[:,chosen_idx[0]]), color="green")
            plt.title(f" Best guess (and uncertainty) at additive contributions from {[problem_info.names[i] for i in chosen_idx]}with sobol index {sobols[idx]}")
            if lim is not None:
                plt.xlim(-lim,lim)
        elif len(chosen_idx)==2:
            mean = model.predict_indiv_mean(x_plot,data,chosen_idx)
            #mean = mean[j:j+1,:] #* (data.Y_std)# + data.Y_mean
            col = plt.scatter(x_plot[:,chosen_idx[0]],x_plot[:,chosen_idx[1]],c=mean)
            #plt.ylim([jnp.min(z[:,chosen_idx[1]]),jnp.max(z[:,chosen_idx[1]])])
            plt.colorbar(col)
            plt.scatter(z[:,chosen_idx[0]],z[:,chosen_idx[1]], color="black",alpha=0.1)
            #ip = model.get_inducing_locations()
            #plt.scatter(ip[:,chosen_idx[0]],ip[:,chosen_idx[1]], color="green")
            plt.title(f"Best guess at additive contribution from {[problem_info.names[i] for i in chosen_idx]} with sobol index {sobols[idx]}")
            # plt.xlim([jnp.min(jnp.hstack([x_plot[:,chosen_idx[0]],ip[:,chosen_idx[0]]])),jnp.max(jnp.hstack([x_plot[:,chosen_idx[0]],ip[:,chosen_idx[0]]]))])
            # plt.ylim([jnp.min(jnp.hstack([x_plot[:,chosen_idx[1]],ip[:,chosen_idx[1]]])),jnp.max(jnp.hstack([x_plot[:,chosen_idx[1]],ip[:,chosen_idx[1]]]))])
            if lim is not None:
                plt.xlim(-lim,lim)  
                plt.ylim(-lim,lim)    
        