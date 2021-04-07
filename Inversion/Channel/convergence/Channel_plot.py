import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import ticker,cm
import sys 
sys.path.append(os.getcwd()+'/Channel')
from forward import Prediction
from simulation import simulation
# from statsmodels.graphics.tsaplots import plot_acf



class Plot_convergence(object):
    def __init__(self):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # training on GPU or CPU
        self.prediction = Prediction('scale_64',device)
        self.simulation = simulation()
        self.true_permeability = np.loadtxt(os.getcwd()+f'/Channel/test_data/true_permeability_0.05.dat').reshape(64,64)
        self.obs = np.loadtxt(os.getcwd()+f'/Channel/test_data/obs_0.05.dat')
        self.sigma = np.loadtxt(os.getcwd()+f'/Channel/test_data/sigma_0.05.dat')
        self.ref_fx,_,_,_,_,_ = self.simulation.demo64(self.true_permeability)
        self.ref_sswr = np.sum(np.power((self.obs-self.ref_fx)/self.sigma,2.0) )
        self.ref_rss = np.sum(np.power((self.obs-self.ref_fx),2.0) )
    
    def compute_para_rss(self,z, exp):
        num  = z.shape[1]
        permeability_samples = np.full((num,4096),0.0)
        para_rss = np.full((num,),0.0)
        if exp=='inversion_64':
            model = self.prediction.model(exp)
            for i in range(num):
                permeability_samples[i,:] = self.prediction.permeability(exp,model,z[:,i]).cpu().numpy().reshape(4096,)
                para_rss[i,] = np.sum(np.power((permeability_samples[i,:]-np.log(self.true_permeability.reshape(4096,))),2))
        if exp=='inversion_16_64':
            model = self.prediction.model(exp)
            for i in range(num):
                zc = z[:16,i]
                zf = z[16:,i]
                permeability_samples[i,:] = self.prediction.permeability(exp,model,zc,zf =zf).cpu().numpy().reshape(4096,)
                para_rss[i,] = np.sum(np.power((permeability_samples[i,:]-np.log(self.true_permeability.reshape(4096,))),2))
        if exp=='inversion_16_32_64':
            model = self.prediction.model(exp)
            for i in range(num):
                zc = z[:80,i]
                zf = z[80:,i]
                permeability_samples[i,:] = self.prediction.permeability(exp,model,zc,zf =zf).cpu().numpy().reshape(4096,)
                para_rss[i,] = np.sum(np.power((permeability_samples[i,:]-np.log(self.true_permeability.reshape(4096,))),2))
        return para_rss,permeability_samples
    
    def plot_para_rss(self,para_rss_64,para_rss_16_64,para_rss_16_32_64):
        fig,ax =plt.subplots()
        # ax.set_yscale('log')
        x_axix=np.linspace(0,30000,30000)
        x_axix1=np.linspace(0,10000,10000)
        x_axix2=np.linspace(0,5000,5000)
        plt.ylabel(r'$SSR_{para}$')
        plt.xlabel('Iterations')
        plt.plot(x_axix, para_rss_64[0:], color='green', label='reference')
        plt.plot(x_axix1, para_rss_16_64[0:], color='blue', label='16-64')
        plt.plot(x_axix2, para_rss_16_32_64[0:], color='red', label='16-32-64')
        plt.legend()
        plt.savefig(os.getcwd()+ f'/Channel/convergence/images/channel_para_ssr_convergence.pdf',bbox_inches = 'tight', dpi=1000,pad_inches=0.0)
        plt.close()
      

    def para_rss_convergence(self):
        z_64 = np.loadtxt(os.getcwd()+f'/Channel/prediction/inversion_64/pCN_step0.08_0.05_30000_28000_all_samples.dat')
        z_16_64 = np.loadtxt(os.getcwd()+f'/Channel/prediction/inversion_16_64_10000/pCN_step0.08_0.05_10000_8000_all_samples.dat')
        z_16_32_64 = np.loadtxt(os.getcwd()+f'/Channel/prediction/inversion_16_32_64/pCN_step0.08_0.05_5000_3000_all_samples.dat')
        para_rss_64,permeability_samples_64 = self.compute_para_rss(z_64 , 'inversion_64')
        para_rss_16_64,permeability_samples_16_64 = self.compute_para_rss(z_16_64, 'inversion_16_64')
        para_rss_16_32_64,permeability_samples_16_32_64 = self.compute_para_rss(z_16_32_64, 'inversion_16_32_64')
        self.plot_para_rss(para_rss_64,para_rss_16_64,para_rss_16_32_64)

    
    def Comp_log_likelihood(self,fx):
        e=self.obs-fx
        rss = np.sum(np.power(e,2.0))
        sswr = np.sum(np.power(e/self.sigma,2.0))
        nsswr = sswr/self.ref_sswr
        nrss = rss/self.ref_rss
        return  rss,nrss, sswr,nsswr

    def plot_obs_rss(self,rss64,rss16_64,rss16_32_64):
        fig,ax =plt.subplots()
        ax.set_yscale('log')
        x_axix=np.linspace(0,30000,30000)
        x_axix1=np.linspace(0,10000,10000)
        x_axix2=np.linspace(0,5000,5000)
        plt.ylabel(r'$SSR_{obs}$')
        plt.xlabel('Iterations')
        plt.plot(x_axix, rss64[0:], color='green', label='reference')
        plt.plot(x_axix1, rss16_64[0:], color='blue', label='16-64')
        plt.plot(x_axix2, rss16_32_64[0:], color='red', label='16-32-64')
        plt.legend()
        plt.savefig(os.getcwd()+ f'/Channel/convergence/images/channel_obs_ssr_convergence.pdf',bbox_inches = 'tight', dpi=1000,pad_inches=0.0)
        plt.close()

    def plot_nsswr(self,nsswr64,nsswr16_64,nsswr16_32_64):
        fig,ax =plt.subplots()
        ax.set_yscale('log')
        x_axix=np.linspace(0,30000,30000)
        x_axix1=np.linspace(0,10000,10000)
        x_axix2=np.linspace(0,5000,5000)
        plt.ylabel(r'$NSSWR$')
        plt.xlabel('Iterations')
        plt.plot(x_axix, nsswr64[0:], color='green', label='reference')
        plt.plot(x_axix1, nsswr16_64[0:], color='blue', label='16-64')
        plt.plot(x_axix2, nsswr16_32_64[0:], color='red', label='16-32-64')
        plt.legend()
        plt.savefig(os.getcwd()+ f'/Channel/convergence/images/channel_nsswr_convergence.pdf',bbox_inches = 'tight', dpi=1000,pad_inches=0.0)
        plt.close()

    def nsswr_convergence(self):
        fx64 = np.loadtxt(os.getcwd()+f'/Channel/prediction/inversion_64/pCN_step0.08_0.05_30000_28000_fx_obs.dat')
        fx16_64 = np.loadtxt(os.getcwd()+f'/Channel/prediction/inversion_16_64_10000/pCN_step0.08_0.05_10000_8000_fx_obs.dat')
        fx16_32_64 = np.loadtxt(os.getcwd()+f'/Channel/prediction/inversion_16_32_64/pCN_step0.08_0.05_5000_3000_fx_obs.dat')
        rss64 = np.full((fx64.shape[1],1),0.0)
        nrss64 = np.full((fx64.shape[1],64),0.0)
        sswr64 = np.full((fx64.shape[1],1),0.0)
        nsswr64 = np.full((fx64.shape[1],1),0.0)
        nrss16_64 = np.full((fx16_64.shape[1],64),0.0)
        rss16_64 = np.full((fx16_64.shape[1],1),0.0)
        sswr16_64 = np.full((fx16_64.shape[1],1),0.0)
        nsswr16_64 = np.full((fx16_64.shape[1],1),0.0)
        nrss16_32_64 = np.full((fx16_32_64.shape[1],64),0.0)
        rss16_32_64 = np.full((fx16_32_64.shape[1],1),0.0)
        sswr16_32_64 = np.full((fx16_32_64.shape[1],1),0.0)
        nsswr16_32_64 = np.full((fx16_32_64.shape[1],1),0.0)
        for i in range(fx64.shape[1]):
            rss64[i,:],nrss64[i,:], sswr64[i,:],nsswr64[i,:]= self.Comp_log_likelihood(fx64[:,i])
        for i in range(fx16_64.shape[1]):   
            rss16_64[i,:],nrss16_64, sswr16_64[i,:],nsswr16_64[i,:]= self.Comp_log_likelihood(fx16_64[:,i])
        for i in range(fx16_32_64.shape[1]):   
            rss16_32_64[i,:],nrss16_32_64, sswr16_32_64[i,:],nsswr16_32_64[i,:] = self.Comp_log_likelihood(fx16_32_64[:,i])
        self.plot_nsswr(nsswr64,nsswr16_64,nsswr16_32_64)
        self.plot_obs_rss(rss64,rss16_64,rss16_32_64)

   

    
    def acf(self,all_samples64,all_samples16_64,all_samples16_32_64):
        samples = np.full((1000,64,64),0.0)
        exp = 'inversion_64'
        model = self.prediction.model(exp)
        zc = all_samples64[:,29000:]
        # zf = all_samples16_64[16:,29000:]
        samples = np.log(self.prediction.permeability(exp,model,zc,zf=None).cpu().numpy().reshape(-1,64,64) )     
        return samples


    def plot_para_state(self):
        all_samples64 = np.loadtxt(os.getcwd()+f'/Channel/prediction/inversion_64/pCN_step0.08_0.05_30000_28000_all_samples.dat')
        all_samples16_64 = np.loadtxt(os.getcwd()+f'/Channel/prediction/inversion_16_64_10000/pCN_step0.08_0.05_10000_8000_all_samples.dat')
        all_samples16_32_64 = np.loadtxt(os.getcwd()+f'/Channel/prediction/inversion_16_32_64/pCN_step0.08_0.05_5000_3000_all_samples.dat')
        samples = np.full((15,64,64),0.0)
        truth = np.log(np.loadtxt(os.getcwd()+f'/Channel/test_data/true_permeability_0.05.dat'))
        latent_sample64 = all_samples64[:,[0,2999,4999,14999,29999]]
        latent_sample16_64 = all_samples16_64[:,[0,999,2999,4999,9999]]
        latent_sample16_32_64 = all_samples16_32_64[:,[0,999,1999,2999,4999]]
        for i in range(15):
            if i<5:
                exp = 'inversion_64'
                model = self.prediction.model(exp)
                samples[i,:,:] = self.prediction.permeability(exp,model,latent_sample64[:,i]).cpu().numpy().reshape(64,64)
            elif 4<i<10:
                exp = 'inversion_16_64'
                model = self.prediction.model(exp)
                zc = latent_sample16_64[:16,i-5]
                zf = latent_sample16_64[16:,i-5]
                samples[i,:,:] = self.prediction.permeability(exp,model,zc,zf=zf).cpu().numpy().reshape(64,64)
            elif 9<i<15:
                exp = 'inversion_16_32_64'
                model = self.prediction.model(exp)
                zc = latent_sample16_32_64[:80,i-10]
                zf = latent_sample16_32_64[80:,i-10]
                samples[i,:,:] = self.prediction.permeability(exp,model,zc,zf=zf).cpu().numpy().reshape(64,64)     
        fig, _ = plt.subplots(3,5, figsize=(15, 9))
        vmin1 = np.amin(truth)
        vmax1 = np.amax(truth)
        for j, ax in enumerate(fig.axes):
            ax.set_aspect('equal')
            ax.set_axis_off()
            cax = ax.imshow(samples[j],  cmap='jet', origin='upper',vmin=vmin1,vmax=vmax1)
            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
            if j == 0:
                ax.set_title('initial state',fontsize=12)
            if j == 1:
                ax.set_title('3000-th state',fontsize=12)
            if j == 2:
                ax.set_title('5000-th state',fontsize=12)
            if j == 3:
                ax.set_title('15000-th state',fontsize=12)
            if j == 4:
                ax.set_title('30000-th state',fontsize=12)
            if j == 5:
                ax.set_title('initial state',fontsize=12)
            if j == 6:
                ax.set_title('1000-th state',fontsize=12)
            if j == 7:
                ax.set_title('3000-th state',fontsize=12)
            if j == 8:
                ax.set_title('5000-th state',fontsize=12)
            if j == 9:
                ax.set_title('10000-th state',fontsize=12)
            if j == 10:
                ax.set_title('initial state',fontsize=12)
            if j == 11:
                ax.set_title('1000-th state',fontsize=12)
            if j == 12:
                ax.set_title('2000-th state',fontsize=12)
            if j == 13:
                ax.set_title('3000-th state',fontsize=12)
            if j == 14:
                ax.set_title('5000-th state',fontsize=12)
        plt.savefig(os.getcwd()+f'/Channel/convergence/images/channel_para_state.pdf',bbox_inches = 'tight', dpi=1000,pad_inches=0.0)


if __name__=='__main__':




    '''
    plot NSSWR and Para_MSE in 64*64 resolution for three experiments
    '''
    plot = Plot_convergence()
    plot.nsswr_convergence()
    plot.para_rss_convergence()
    plot.plot_para_state()



    '''
    plot Morkov chain states in 64*64 resolution for three experiments
    '''

    # samples = plot.acf(all_samples64,all_samples16_64,all_samples16_32_64)
    # a = samples[:,10,10].reshape(-1,)
    # plot_acf(a,lags=300)
    # plt.savefig(os.getcwd()+'/acf64.pdf')
    # plt.close()







    