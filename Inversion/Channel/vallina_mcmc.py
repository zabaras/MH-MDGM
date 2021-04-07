import numpy as np 
from forward import Prediction
import matplotlib.pyplot as plt
import os
import scipy
from matplotlib import ticker,cm

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
        self.Prior_type = 'Standard Normal'
        self.plot_interval = 1000
        self.num_obs = 64
        self.prediction = Prediction(scale,device)

    def pCN(self,old_params,beta):
        '''
        Here the prior distribution is standart Normal
        '''
        new_params = np.sqrt(1-beta*beta)*old_params+beta*np.random.randn(self.dim,)
        return new_params

    def fx(self,model,exp,z): 
        '''
        compute the permeability field X using MDGM, and compute f(X) using forward model
        '''  
        input = self.prediction.permeability(exp,model,z)
        like , pre = self.prediction.forward(input)
        fx = np.reshape(like,(1,-1))
        return fx
  
    def Comp_log_likelihood(self,z,obs,sigma,exp,model):
        '''
        compute likelihood function for inference
        '''
        fx = self.fx(model,exp,z)
        e=obs-fx
        log_likelihood = -0.5 * np.sum( np.power(e/sigma,2.0))
        return fx, log_likelihood
    
    def pCN_MH(self,init_state,beta,exp,noise,output_dir):
        '''
        pCN MH refer to 'Variational data assimilation using targetted random walks'
        and we have '-' in the log-likelihood
        MH algorithm with DGM on the first scale(the coarsest scale)
        proposal for latent variable z1 with big step size to explore the whole space
        (first 50% state using 0.08, rest with 0.04)
        '''
        model = self.prediction.model(exp)
        accept_num = 0
        old_params = init_state
        samples = np.zeros([self.dim,self.step])
        log_likelihood = np.zeros([1,self.step])
        fx_obs = np.zeros([self.num_obs,self.step])
        old_fx ,old_log_l= self.Comp_log_likelihood(old_params,self.obs,self.sigma,exp,model)
        fx_obs[:,0] = old_fx
        log_likelihood[:,0] = old_log_l
        samples [:,0] = old_params.reshape([self.dim,])
        self.plot_para_field(0,old_params,self.true_permeability, self.true_pressure,beta,'pCN',exp,model,noise,output_dir) 
        for i in range(1,self.step):
            if i > 0.5*self.step:
               beta = 0.04
            new_params = self.pCN(old_params,beta)
            new_fx ,new_log_l = self.Comp_log_likelihood(new_params,self.obs,self.sigma,exp,model)
            ratio = np.exp(new_log_l - old_log_l)
            alpha = min(1,ratio)
            np.random.seed(i)
            z = np.random.rand()
            if z<= alpha:
                print('-----------------------------------------------------------------')
                old_params = new_params.reshape([self.dim,])
                fx_obs[:,i] = new_fx
                log_likelihood[:,i] = new_log_l
                samples[:,i] = new_params.reshape([self.dim,])
                old_log_l = new_log_l
                old_fx = new_fx
                accept_num = accept_num +1
            elif z>alpha:
                old_params = old_params
                samples[:,i] = old_params.reshape([self.dim,])
                fx_obs[:,i] = old_fx
                log_likelihood[:,i] = old_log_l
            if (i+1)%self.plot_interval == 0:
                self.plot_para_field(i,old_params,self.true_permeability, self.true_pressure,beta,'pCN',exp,model,noise,output_dir)
        post_samples = samples[:,self.burn_in:]
        accept_ratio = accept_num/self.step
        print('accept_ratio:',accept_ratio)
        accept = [accept_ratio,accept_num]
        return samples, post_samples,fx_obs,accept,log_likelihood

    def plot_para_field(self,i,params, true_permeability, true_pressure,beta,type,exp,model,noise,output_dir):
        '''
        Plot Markov chain state(log permeability)
        '''
        pre_permeability =  self.prediction.permeability(exp,model,params)
        _, pre_pressure = self.prediction.forward(pre_permeability)
        if exp=='inversion_16':
            pre_permeability = pre_permeability.data.cpu().numpy().reshape(16,16)
        if exp=='inversion_64':
            pre_permeability = pre_permeability.data.cpu().numpy().reshape(64,64)
        samples = [np.log(true_permeability),pre_permeability, true_pressure,pre_pressure]
        fig, _ = plt.subplots(2,2, figsize=(6,6))
        vmin1 = [np.amin(samples[0]), np.amin(samples[0]),np.amin(samples[2]),np.amin(samples[2])]
        vmax1 = [np.amax(samples[0]), np.amax(samples[0]),np.amax(samples[2]),np.amax(samples[2])]
        for j, ax in enumerate(fig.axes):
            ax.set_aspect('equal')
            ax.set_axis_off()
            cax = ax.imshow(samples[j],  cmap='jet',vmin=vmin1[j],vmax=vmax1[j])
            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
        plt.savefig(output_dir+f'/{type}_step_size{beta}_state_{i+1}.pdf')
        plt.close()