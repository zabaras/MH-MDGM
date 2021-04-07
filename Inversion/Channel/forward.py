from simulation import simulation
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import os
import scipy
import numpy as np
from CONV_VAE_model import Decoder,Encoder

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
        self.simulation = simulation()
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

        if exp=='inversion_64':
            # model_dir = dir+f'/Channel/generative_model/AAEdecoder24_bench64_0.01_epoch50.pth'
            model_dir = dir+f'/Channel/generative_model/decoder_64_VAE_0.5_epoch50.pth'
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
            zc = (torch.FloatTensor(zc)).to(self.device).reshape(1,1,4,4)
            with torch.no_grad():
                input = model(zc).reshape(16,16)

        if exp=='inversion_64':
            zc = (torch.FloatTensor(zc)).to(self.device).reshape(1,1,16,16)
            with torch.no_grad():
                input = model(zc).reshape(1,1,64,64)

        if exp=='inversion_16_64':
            zf = zf.reshape(1,1,16,16)
            zf = (torch.FloatTensor(zf)).to(self.device)
            zc = (torch.FloatTensor(zc)).to(self.device).reshape(1,1,4,4)
            zc= self.upsample(zc,'16_64')
            z_h = torch.cat((zc,zf),1)
            with torch.no_grad():
                input = model(z_h).reshape(64,64)

        if exp=='inversion_16_32':
            zf = zf.reshape(1,1,8,8)
            zf = (torch.FloatTensor(zf)).to(self.device)
            zc = (torch.FloatTensor(zc)).to(self.device).reshape(1,1,4,4)
            zc= self.upsample(zc,'16_32')
            z_h = torch.cat((zc,zf),1)
            with torch.no_grad():
                input = model(z_h).reshape(32,32)

        if exp=='inversion_16_32_64':
            zf = zf.reshape(16,16)
            zf = (torch.FloatTensor(zf)).to(self.device).reshape(1,1,16,16)
            z16 = zc[:16].reshape(4,4)
            z16 = (torch.FloatTensor(z16)).to(self.device).reshape(1,1,4,4)
            z32 = zc[16:].reshape(8,8)
            z32 = (torch.FloatTensor(z32)).to(self.device).reshape(1,1,8,8)
            z16= self.upsample(z16,'16_64')
            z32= self.upsample(z32,'32_64')
            z_h = torch.cat((z16,z32,zf),1)
            with torch.no_grad():
                input = model(z_h).reshape(64,64)
        return input


    def forward(self,input):
        if self.scale == 'scale_16':
            input=input.data.cpu().numpy()
            like,pre,_,_,_,_ = self.simulation.demo16(np.exp(input))
            

        elif self.scale == 'scale_32':
            input=input.data.cpu().numpy()
            like,pre,_,_,_,_ = self.simulation.demo32(np.exp(input))

        elif self.scale == 'scale_64':
            input=input.data.cpu().numpy()
            like,pre,_,_,_,_ = self.simulation.demo64(np.exp(input))

        return like , pre

  
