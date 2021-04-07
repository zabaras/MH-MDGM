import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import os
import scipy
import numpy as np
from matplotlib import ticker,cm
from forward import Prediction
from CONV_VAE_model import Decoder,Encoder
from matplotlib import ticker,cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class mean_std_plot(object):
    def __init__(self,device,scale_size,exp,noise):
        self.device= device
        self.scale = scale_size
        self.prediction = Prediction(self.scale,self.device)
        self.exp = exp
        self.noise = noise
    def compute_mean_std(self,pos,truth,save_dir):
        model = self.prediction.model(self.exp)
        num  = pos.shape[1]
        pos_samples = np.full([num,self.scale*self.scale],0.0)
        permeability_samples = np.full([num,self.scale,self.scale],0.0)
        if self.exp =='inversion_16' or self.exp =='inversion_32' or self.exp =='inversion_64':
            zc = pos 
            for i in range(num):
                pos_samples[i,:] = self.prediction.permeability(self.exp,model,zc[:,i],zf=None).cpu().numpy().reshape(self.scale*self.scale,)
                permeability_samples[i,:,:]=pos_samples[i,:].reshape(self.scale,self.scale)
        if self.exp =='inversion_16_64':
            zc = pos[:16,:]
            zf = pos[16:,:]
            for i in range(num):
                pos_samples[i,:] = self.prediction.permeability(self.exp,model,zc[:,i],zf=zf[:,i]).cpu().numpy().reshape(self.scale*self.scale,)
                permeability_samples[i,:,:]=pos_samples[i,:].reshape(self.scale,self.scale)
        mean = np.mean(permeability_samples,axis = 0)
        std = np.std(permeability_samples,axis = 0)
        sample1 = permeability_samples[0,:,:].reshape(self.scale,self.scale)
        sample2 = permeability_samples[1000,:,:].reshape(self.scale,self.scale)
        sample3 = permeability_samples[1999,:,:].reshape(self.scale,self.scale)
        #np.savetxt(save_dir+f'/pos_samples.dat',pos_samples)
        np.savetxt(save_dir+f'/mean.dat',mean)
        np.savetxt(save_dir+f'/std.dat',std)
        self.plot(truth,mean,std,sample1,sample2,sample3,save_dir)
        return pos_samples

    def plot(self,truth,mean,std,sample1,sample2,sample3,save_dir):
        samples = [np.log(truth),mean,std,sample1,sample2,sample3]
        fig, _ = plt.subplots(2,3, figsize=(9, 6))
        vmin1 = [np.amin(samples[0]), np.amin(samples[0]),np.amin(samples[2]),np.amin(samples[0]),np.amin(samples[0]),np.amin(samples[0])]
        vmax1 = [np.amax(samples[0]), np.amax(samples[0]),np.amax(samples[2]),np.amax(samples[0]),np.amax(samples[0]),np.amax(samples[0])]
        for j, ax in enumerate(fig.axes):
            ax.set_aspect('equal')
            ax.set_axis_off()
            cax = ax.imshow(samples[j],  cmap='jet', origin='upper',vmin=vmin1[j],vmax=vmax1[j])
            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
            if j == 0:
                ax.set_title('Exact',fontsize=12)
            if j == 1:
                ax.set_title('Mean',fontsize=12)
            if j == 2:
                ax.set_title('Standard deviation',fontsize=12)
            if j == 3:
                ax.set_title('Posterior sample 1',fontsize=12)
            if j == 4:
                ax.set_title('Posterior sample 2',fontsize=12)
            if j == 5:
                ax.set_title('Posterior sample 3',fontsize=12)
        plt.savefig(save_dir+f'/gaussian_statistics_{self.exp}.pdf',bbox_inches = 'tight', dpi=1000,pad_inches=0.0)
        plt.close()


    
