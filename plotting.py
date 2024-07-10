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




def plot_interactions(problem_info:ProblemInfo, model, data, k=10,use_range=False, greedy=False, lim=None, bij_list_of_lists=None):
        
    
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
    ip = model.get_inducing_locations()
    zmax = jnp.max(ip, axis=0)
    zmin = jnp.min(ip, axis=0)


    
    for j in range(model.num_latents):
        plt.figure()
        plt.plot(sobols_for_plot[j])
        plt.title(f"latent{j} has sobol indicies (red lines between orders)")
        plt.axvline(x=0, color="red")
        plt.axvline(x=problem_info.num_variables, color="red")
    
    
        for i, idx in enumerate(jax.lax.top_k(sobols[j], k)[1]):
            chosen_idx = idxs[idx]
            plt.figure()
            num_plot = 1_000 if len(chosen_idx)==1 else 5_000
            from scipy.stats import qmc
            sampler = qmc.Halton(d=problem_info.num_variables)
            x_plot = sampler.random(n=num_plot)* (zmax - zmin) + zmin
            if len(chosen_idx)==1:     
                try:
                    mean, var = model.predict_indiv(x_plot,data, chosen_idx)
                except:
                    mean, var = model.predict_indiv(x_plot,chosen_idx)
                std = jnp.sqrt(var)
                mean = mean[:,j:j+1] #* (data.Y_std)#+ data.Y_mean
                std = std[:,j:j+1]#* (data.Y_std)
                ip = model.get_inducing_locations()[:,chosen_idx[0]:chosen_idx[0]+1]
                x_plot_1 = x_plot[:,chosen_idx[0]:chosen_idx[0]+1]
                z_1 = z[:,chosen_idx[0]:chosen_idx[0]+1]
                if bij_list_of_lists is not None:
                    for bij in bij_list_of_lists[chosen_idx[0]][::-1]:
                        ip = bij.inverse(ip)
                        x_plot_1 = bij.inverse(x_plot_1)
                        z_1 = bij.inverse(z_1)
                
                
                
                
                plt.scatter(x_plot_1,mean, color="blue") 
                plt.scatter(x_plot_1,mean+ 1.96*std, color="red") 
                plt.scatter(x_plot_1,mean- 1.96*std, color="red") 
                # plt.scatter(z_1,jnp.zeros_like(z_1), color="black",alpha=0.01)
                # plt.scatter(ip,jnp.zeros_like(ip), color="orange")
                plt.title(f"Latent {j} with rank {i}: Best guess (and uncertainty) at additive contributions from {[problem_info.names[i] for i in chosen_idx]}with sobol index {sobols[j][idx]}")
                if lim is not None:
                    plt.xlim(-lim,lim)
                    # plt.ylim(-1.5,3.0)
                else:
                    plt.xlim(jnp.min(ip), jnp.max(ip))
            elif len(chosen_idx)==2:
                try:
                    mean, _ = model.predict_indiv(x_plot,data,chosen_idx)
                except:
                    mean, _ = model.predict_indiv(x_plot,chosen_idx)
                mean = mean[:, j:j+1] #* (data.Y_std)# + data.Y_mean
                

                ip_1 = model.get_inducing_locations()[:,chosen_idx[0]:chosen_idx[0]+1]
                ip_2 = model.get_inducing_locations()[:,chosen_idx[1]:chosen_idx[1]+1]
                x_plot_1 = x_plot[:,chosen_idx[0]:chosen_idx[0]+1]
                x_plot_2 = x_plot[:,chosen_idx[1]:chosen_idx[1]+1]
                z_1 = z[:,chosen_idx[0]:chosen_idx[0]+1]
                z_2 = z[:,chosen_idx[1]:chosen_idx[1]+1]
                if bij_list_of_lists is not None:
                    for bij in bij_list_of_lists[chosen_idx[0]][::-1]:
                        ip_1 = bij.inverse(ip_1)
                        x_plot_1 = bij.inverse(x_plot_1)
                    for bij in bij_list_of_lists[chosen_idx[1]][::-1]:
                        ip_2 = bij.inverse(ip_2)
                        x_plot_2 = bij.inverse(x_plot_2)
                
                col = plt.scatter(x_plot_1,x_plot_2,c=mean)
                plt.colorbar(col)
                #plt.scatter(z_1,z_2, color="black",alpha=0.01)
                # plt.scatter(ip_1,ip_2, color="orange")
                plt.title(f"Latent {j} with rank {i}: Best guess at additive contribution from {[problem_info.names[i] for i in chosen_idx]} with sobol index {sobols[j][idx]}")
                if lim is not None:
                    plt.xlim(-lim,lim)  
                    plt.ylim(-lim,lim)  
                else:
                    plt.xlim(jnp.min(ip_1), jnp.max(ip_1)) 
                    plt.ylim(jnp.min(ip_2), jnp.max(ip_2))  
        