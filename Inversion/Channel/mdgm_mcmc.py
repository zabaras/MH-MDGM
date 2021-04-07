import numpy as np 
from forward import Prediction
import matplotlib.pyplot as plt
import os
import scipy
from matplotlib import ticker,cm
from latent import latent_c

class inference(object):
    def __init__(self,step,burn_in,dim,obs,sigma,true_permeability,true_pressure,obs_position,scale,device):
        self.dim = dim
        self.burn_in = burn_in
        self.step = step
        self.obs = obs
        self.sigma = sigma
        self.true_permeability = true_permeability
        self.true_pressure = true_pressure
        self.pos = obs_position
        self.plot_interval = 1000
        self.num_obs = 64
        self.prediction = Prediction(scale,device)

    def pCN(self,old_params,beta,dim):
        '''
        Here the prior distribution is standart Normal
        '''
        new_params = np.sqrt(1-beta*beta)*old_params+beta*np.random.randn(dim,)
        return new_params

    def fx(self,model,exp,zc,zf):  
        '''
        compute the permeability field X using MDGM, and compute f(X) using forward model
        ''' 
        input = self.prediction.permeability(exp,model,zc,zf=zf)
        like , pre = self.prediction.forward(input)
        f_x = np.reshape(like,(1,-1))
        return f_x,input,pre  

    def Comp_log_likelihood(self,obs,sigma,exp,model,zc,zf=None):
        '''
        compute likelihood function for inference
        '''
        f_x,input,pre = self.fx(model,exp,zc,zf)
        e=obs-f_x
        log_likelihood = -0.5 * np.sum( np.power(e/sigma,2.0))
        return f_x,input,pre, log_likelihood
    
    def pCN_MH(self,init_c,init_state_f,beta,exp,noise,output_dir):
        '''
        pCN MH refer to 'Variational data assimilation using targetted random walks'
        and we have '-' in the log-likelihood
        MH algorithm with MDGM, two proposal distributions for latent zc and zf respectively,
        where zc can capture global feature, zf is latent variable for parameter local adaption.
        proposal for zc with small step size(0.01)
        proposal for zf with big step size(first 50% state using 0.08, rest with 0.04)
        '''
        coarse_beta = 0.01
        model = self.prediction.model(exp)
        old_params_c = init_c
        dim_c=old_params_c.shape[0]
        accept_num = 0
        dim = dim_c+self.dim
        samples = np.zeros([dim,self.step])
        fx_obs = np.zeros([self.num_obs,self.step])
        log_likelihood = np.zeros([1,self.step])
        old_params_f = init_state_f
        old_fx,old_input,old_pre,old_log_l= self.Comp_log_likelihood(self.obs,self.sigma,exp,model,old_params_c,zf = old_params_f)
        old_params = np.concatenate((old_params_c,old_params_f),axis=0)
        samples[:,0] = old_params.reshape([dim,])
        fx_obs[:,0] = old_fx
        log_likelihood[:,0] = old_log_l
        self.plot_para_field(0,old_input,old_pre,self.true_permeability, self.true_pressure,beta,'pCN',exp,noise,output_dir)
        for i in range(1,self.step):
            if i > 0.5*self.step:
               beta = 0.04
            new_params_c = self.pCN(old_params_c,coarse_beta,dim_c)
            new_params_f = self.pCN(old_params_f,beta,self.dim)
            new_fx,new_input, new_pre, new_log_l= self.Comp_log_likelihood(self.obs,self.sigma,exp,model,new_params_c,zf = new_params_f)            
            new_params = np.concatenate((new_params_c,new_params_f),axis=0)
            ratio2 = np.exp(new_log_l - old_log_l)
            alpha = min(1,ratio2)
            np.random.seed(i)
            z = np.random.rand()
            if z<= alpha:
                print('-----------------------------------------------------------------')
                log_likelihood[:,i] = new_log_l
                fx_obs[:,i] = new_fx
                old_fx= new_fx
                samples[:,i] = new_params
                old_input = new_input
                old_pre = new_pre
                old_params = new_params
                old_params_c = new_params_c 
                old_params_f = new_params_f
                old_log_l = new_log_l
                accept_num = accept_num +1
            elif z>alpha:
                samples[:,i] = old_params
                fx_obs[:,i] = old_fx
                log_likelihood[:,i] = old_log_l
            if  (i+1)%self.plot_interval == 0:
                self.plot_para_field(i,old_input,old_pre,self.true_permeability, self.true_pressure,beta,'pCN',exp,noise,output_dir)

        post_samples = samples[:,self.burn_in:]
        accept_ratio = (accept_num/self.step)
        print('accept_ratio:',accept_ratio)
        accept = [accept_ratio,accept_num]
        return samples, post_samples,fx_obs,accept,log_likelihood
     

    def plot_para_field(self,i,input,pre_pressure, true_permeability, true_pressure,beta,type,exp,noise,output_dir):
        '''
        Plot Markov chain state(log permeability)
        '''
        samples = [np.log(true_permeability),input.data.cpu().numpy(), true_pressure.reshape(64,64),pre_pressure]
        fig, _ = plt.subplots(2,2, figsize=(6,6))
        vmin1 = [np.amin(samples[0]), np.amin(samples[0]),np.amin(samples[2]), np.amin(samples[2])]
        vmax1 = [np.amax(samples[0]), np.amax(samples[0]),np.amax(samples[2]), np.amax(samples[2])]   
        for j, ax in enumerate(fig.axes):
            ax.set_aspect('equal')
            ax.set_axis_off()
            cax = ax.imshow(samples[j],  cmap='jet',vmin=vmin1[j],vmax=vmax1[j])
            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
        plt.savefig(output_dir+f'/{type}_step_size{beta}_state_{i+1}.pdf')
        plt.close()