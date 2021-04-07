import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import os
import scipy
import numpy as np
from CONV_VAE_model import Decoder,Encoder
from forward import Prediction
from latent import latent_c
from matplotlib import ticker,cm

class upsample(nn.Module):
    def __init__(self):
        super(upsample, self).__init__()
        self.transup = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
    def forward(self,zl,size):
        if size == '16_32':
            zl = self.transup(zl)
        if size == '16_64':
            zl = self.transup(zl)
            zl = self.transup(zl)
        if size == '32_64':
            zl = self.transup(zl)
        return zl

class Prediction(object):
    def __init__(self,scale,device):
        self.device = device
        self.scale = scale
        self.upsample = upsample()
        self.upsample.to(self.device)
        

    def model(self,exp):
        dir = os.getcwd()
        if exp=='inversion_16':
            model_dir = dir+f'/Gaussian/generative_model/decoder_16_VAE_0.5_epoch30.pth'
            model = Decoder()
            model.load_state_dict(torch.load(model_dir, map_location=self.device),False)
            model =model.to(self.device)
            model.eval()
        

        if exp=='inversion_16_64':
            model_dir = dir+f'/Gaussian/generative_model/decoder_16_64_VAE_0.5_0.5_epoch30.pth'
            model = Decoder(input_channels=2)
            model.load_state_dict(torch.load(model_dir, map_location=self.device),False)
            model =model.to(self.device)
            model.eval()
        return model


    def permeability(self,exp,model,zc,zf=None):
        if exp=='inversion_16':
            zc = (torch.FloatTensor(zc)).to(self.device).reshape(-1,1,4,4)
            with torch.no_grad():
                input = model(zc).reshape(-1,16,16)
        
        if exp=='inversion_16_64':
            zf = zf.reshape(-1,1,16,16)
            zf = (torch.FloatTensor(zf)).to(self.device)
            zc = (torch.FloatTensor(zc)).to(self.device).reshape(-1,1,4,4)
            zc= self.upsample(zc,'16_64')
            z_h = torch.cat((zc,zf),1)
            with torch.no_grad():
                input = model(z_h).reshape(-1,64,64)
        return input

    def plot_ite(self,samples,filename):
        fig, _ = plt.subplots(2,4, figsize=(12,6))
        vmin1 = 2.5
        vmax1 = -0.5
        for j, ax in enumerate(fig.axes):
            ax.set_aspect('equal')
            ax.set_axis_off()
            cax = ax.imshow(samples[j],  cmap='jet',vmin=vmin1,vmax=vmax1)
            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
        output_dir = os.getcwd()+ f'/Gaussian/MDGM_plot'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir+ filename,bbox_inches = 'tight', dpi=1000,pad_inches=0.0)
        plt.close()

if __name__ == "__main__":
    torch.random.manual_seed(10)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 
    latent_c = latent_c(device)
    prediction = Prediction('scale_64',device)
    nz_x, nz_y = 4, 4
    z = torch.randn(4, 1, nz_x, nz_y)
    model = prediction.model('inversion_16')
    gen_imgs = prediction.permeability('inversion_16',model,z)
    samples = np.squeeze(gen_imgs.data.cpu().numpy())
    condition = np.loadtxt(os.getcwd()+f'/Gaussian/MDGM_plot/gaussian_16.dat')
    # condition = samples[-1]
    zc =latent_c.coarse_latent('inversion_16_64',condition).reshape(16,)
    zc = (np.ones([3,1,4,4]))*zc.reshape(4,4)
    z64 = torch.randn(3, 1, 16, 16)
    model = prediction.model('inversion_16_64')
    gen_imgs1 = prediction.permeability('inversion_16_64',model,zc,zf = z64)
    samples1 = np.squeeze(gen_imgs1.data.cpu().numpy())
    samples2 = [samples[0],samples[1],samples[2],samples[3],condition,samples1[0],samples1[1],samples1[2]]
    filename = '/gaussian_MDGM_16_64.pdf'
    prediction.plot_ite(samples2,filename)





    

    



  


