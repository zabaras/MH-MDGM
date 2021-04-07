import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from CONV_VAE_model import Decoder,Encoder
from forward import Prediction


class latent_c(object):
    def __init__(self,device):
        self.device = device
        self.prediction = Prediction(64,self.device)
        dir = os.getcwd()
        model_dir16 = dir+f'/Channel/generative_model/encoder_16_VAE_1_epoch30.pth'
        model16 = Encoder()
        model16.load_state_dict(torch.load(model_dir16, map_location=self.device),False)
        self.encoder_model16 = model16.to(self.device)
        self.encoder_model16.eval()

        dir = os.getcwd()
        model_dir_16_32 = dir+f'/Channel/generative_model/encoder_16_32_VAE_1_0.7_epoch50.pth'
        model_16_32 = Encoder()
        model_16_32.load_state_dict(torch.load(model_dir_16_32, map_location=self.device),False)
        self.encoder_model_16_32 = model_16_32.to(self.device)
        self.encoder_model_16_32.eval()

    def coarse16(self,x16):
            with torch.no_grad():
                x16 = x16.reshape(1,1,16,16)
                z, mu, logvar = self.encoder_model16(x16)
            return z, mu, logvar

    def coarse16_32(self,x32):
            with torch.no_grad():
                x32 = x32.reshape(1,1,32,32)
                z, mu, logvar = self.encoder_model_16_32(x32)
            return z, mu, logvar



    def upsampling(self,img,factor):
        (x,y) = img.shape
        x_new = int(np.ceil(x/factor))
        y_new = int(np.ceil(y/factor))
        img_new = np.full((x_new,y_new),0.0)
        img = np.exp(img)
        for i in range(x_new):
            for j in range(y_new):
                if i == x_new and j < y_new:
                    mesh = img[i*factor : x, j*factor : (j+1)*factor] 
                elif i < x_new and j == y_new:
                    mesh = img[i*factor : (i+1)*factor, j*factor : y]
                elif i == x_new and j == y_new:
                    mesh = img[i*factor : x, j*factor : y] 
                else:
                    mesh = img[i*factor : (i+1)*factor, j*factor : (j+1)*factor]
                img_new[i,j] = np.mean(mesh)
        img_new = np.log(img_new)
        return img_new

    def coarse_latent(self,exp,coarse_mean):
        if exp=='inversion_16_64' or exp =='inversion_16_32' :
            model = self.prediction.model('inversion_16')
            latent = np.full((1,16),0.0)
            _, mu16,_ = self.coarse16(torch.FloatTensor(coarse_mean).reshape(1,1,16,16).to(self.device))
            latent=mu16.data.cpu().numpy().reshape(16,)
        
        if exp=='inversion_16_32_64':
            model = self.prediction.model('inversion_16_32')
            latent = np.full((1,80),0.0)
            _, mu32,_ = self.coarse16_32(torch.FloatTensor(coarse_mean).reshape(1,1,32,32).to(self.device))
            image16 = self.upsampling(coarse_mean.reshape(32,32),2)
            image16 = (torch.FloatTensor(image16)).to(self.device)
            _, mu16,_ = self.coarse16(image16)
            latent[0,:16]=mu16.data.cpu().numpy().reshape(16,)
            latent[0,16:]=mu32.data.cpu().numpy().reshape(64,)
        return latent




