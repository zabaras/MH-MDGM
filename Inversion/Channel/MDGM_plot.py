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
            model_dir = dir+f'/Channel/generative_model/decoder_16_VAE_1_epoch30.pth'
            model = Decoder()
            model.load_state_dict(torch.load(model_dir, map_location=self.device),False)
            model =model.to(self.device)
            model.eval()


        if exp=='inversion_16_64':
            model_dir = dir+f'/Channel/generative_model/decoder_16_64_VAE_2.5_epoch50.pth'
            model = Decoder(input_channels=2)
            model.load_state_dict(torch.load(model_dir, map_location=self.device),False)
            model =model.to(self.device)
            model.eval()

        if exp=='inversion_16_32':
            model_dir = dir+f'/Channel/generative_model/decoder_16_32_VAE_1_0.7_epoch50.pth'
            model = Decoder(input_channels=2)
            model.load_state_dict(torch.load(model_dir, map_location=self.device),False)
            model =model.to(self.device)
            model.eval()

        if exp=='inversion_16_32_64':
            model_dir = dir+f'/Channel/generative_model/decoder_16_32_64_VAE_1_0.7_0.7_epoch50.pth'
            model = Decoder(input_channels=3)
            model.load_state_dict(torch.load(model_dir, map_location=self.device),False)
            model =model.to(self.device)
            model.eval()
        return model


    def permeability(self,exp,model,zc,zf=None):
        if exp=='inversion_16':
            zc = (torch.FloatTensor(zc)).to(self.device).reshape(-1,1,4,4)
            with torch.no_grad():
                input = model(zc).reshape(-1,16,16)
        
        if exp=='inversion_16_32':
            zf = zf.reshape(-1,1,8,8)
            zf = (torch.FloatTensor(zf)).to(self.device)
            zc = (torch.FloatTensor(zc)).to(self.device).reshape(-1,1,4,4)
            zc= self.upsample(zc,'16_32')
            z_h = torch.cat((zc,zf),1)
            with torch.no_grad():
                input = model(z_h).reshape(-1,32,32)
        
        if exp=='inversion_16_64':
            zf = zf.reshape(-1,1,16,16)
            zf = (torch.FloatTensor(zf)).to(self.device)
            zc = (torch.FloatTensor(zc)).to(self.device).reshape(-1,1,4,4)
            zc= self.upsample(zc,'16_64')
            z_h = torch.cat((zc,zf),1)
            with torch.no_grad():
                input = model(z_h).reshape(-1,64,64)
        if exp=='inversion_16_32_64':
            zf = zf.reshape(-1,1,16,16)
            zf = (torch.FloatTensor(zf)).to(self.device)
            z16 = zc[:,:16].reshape(-1,1,4,4)
            z16 = (torch.FloatTensor(z16)).to(self.device)
            z32 = zc[:,16:].reshape(-1,1,8,8)
            z32 = (torch.FloatTensor(z32)).to(self.device)
            z16= self.upsample(z16,'16_64')
            z32= self.upsample(z32,'32_64')
            z_h = torch.cat((z16,z32,zf),1)
            with torch.no_grad():
                input = model(z_h).reshape(-1,64,64)
        return input

    def plot_ite(self,samples,filename):
        fig, _ = plt.subplots(2,4, figsize=(12,6))
        vmin1 = 5
        vmax1 = -2
        for j, ax in enumerate(fig.axes):
            ax.set_aspect('equal')
            ax.set_axis_off()
            cax = ax.imshow(samples[j],  cmap='jet',vmin=vmin1,vmax=vmax1)
            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
        output_dir = os.getcwd()+ f'/Channel/MDGM_plot'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir+ filename,bbox_inches = 'tight', dpi=1000,pad_inches=0.0)
        plt.close()
if __name__ == "__main__":
   
    '''
    plot MDGM with 2 scales(16-64),16 is single scale(vanilla VAE)
    '''
    torch.random.manual_seed(9)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 
    latent_c = latent_c(device)
    prediction = Prediction('scale_64',device)
    nz_x, nz_y = 4, 4
    z = torch.randn(4, 1, nz_x, nz_y)
    model = prediction.model('inversion_16')
    gen_imgs = prediction.permeability('inversion_16',model,z)
    samples = np.squeeze(gen_imgs.data.cpu().numpy())
    img = np.loadtxt(os.getcwd()+f'/Channel/MDGM_plot/channel_16.dat')
    zc =latent_c.coarse_latent('inversion_16_64',img).reshape(16,)
    zc = (np.ones([3,1,4,4]))*zc.reshape(4,4)
    z64 = torch.randn(3, 1, 16, 16)
    model = prediction.model('inversion_16_64')
    gen_imgs1 = prediction.permeability('inversion_16_64',model,zc,zf = z64)
    samples1 = np.squeeze(gen_imgs1.data.cpu().numpy())
    samples2 = [samples[0],samples[1],samples[2],samples[3],img,samples1[0],samples1[1],samples1[2]]
    filename = '/channel_MDGM_16_64.pdf'
    prediction.plot_ite(samples2,filename)

    '''
    plot MDGM with 3 scales(16-32-64), 16 is single scale(vanilla VAE) present above
    '''

    prediction = Prediction('scale_64',device)
    img16 = np.loadtxt(os.getcwd()+f'/Channel/MDGM_plot/channel_16.dat')
    zc =latent_c.coarse_latent('inversion_16_32',img16).reshape(16,)
    zc = (np.ones([3,16]))*zc.reshape(16,)
    z32 = torch.randn(3, 1, 8, 8)
    model = prediction.model('inversion_16_32')
    gen_imgs1 = prediction.permeability('inversion_16_32',model,zc,zf = z32)
    samples2 = np.squeeze(gen_imgs1.data.cpu().numpy())
    img32 = np.loadtxt(os.getcwd()+f'/Channel/MDGM_plot/channel_32.dat')
    zc =latent_c.coarse_latent('inversion_16_32_64',img32)
    zc = (np.ones([3,80]))*zc.reshape(80,)
    z64 = torch.randn(3, 1, 16, 16)
    model = prediction.model('inversion_16_32_64')
    gen_imgs1 = prediction.permeability('inversion_16_32_64',model,zc,zf = z64)
    samples3 = np.squeeze(gen_imgs1.data.cpu().numpy())
    samples4 = [img16,samples2[0],samples2[1],samples2[2],img32,samples3[0],samples3[1],samples3[2]]
    filename = '/channel_MDGM_16_32_64.pdf'
    prediction.plot_ite(samples4,filename)




    

    



  


