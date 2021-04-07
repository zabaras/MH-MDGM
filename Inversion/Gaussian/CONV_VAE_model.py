import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    eps = torch.randn_like(std)
    return mu + eps*std

class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.BatchNorm2d(in_features)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self,filters,  res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )
    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class Encoder(nn.Module):
    def __init__(self, input_channels=1):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(input_channels, 48, kernel_size=3, stride=2, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(48) for _ in range(1)])
        self.trans = nn.Sequential(
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 24, kernel_size=3, stride=2, padding=1),
        )
        self.main = nn.Sequential(
            nn.BatchNorm2d(24),
            nn.ReLU( inplace=True),
            nn.Conv2d(24, 12,kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(12),
            nn.ReLU( inplace=True),
            nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.mu = nn.Sequential(
            nn.BatchNorm2d(6),
            nn.ReLU( inplace=True),
            nn.Conv2d(6,1, 3, 1, 1, bias=False))
        self.logvar = nn.Sequential(
            nn.BatchNorm2d(6),
            nn.ReLU( inplace=True),
            nn.Conv2d(6,1, 3, 1, 1, bias=False))
       
    def forward(self, img):
        out1 = self.conv(img)         
        out2 = self.res_blocks(out1)   
        out3 = self.trans(out2)     
        out6 = self.main(out3)        
        mu, logvar = self.mu(out6), self.logvar(out6)
        z = reparameterization(mu, logvar) 
        return z, mu, logvar 

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

class Decoder(nn.Module):
    def __init__(self, num_upsample=2, input_channels=1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=3, stride=1, padding=1)
        self.res_block1 = nn.Sequential(*[ResidualInResidualDenseBlock(24) for _ in range(2)])
        self.transup1 = nn.Sequential(
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
        )
        self.res_block2 = nn.Sequential(*[ResidualInResidualDenseBlock(48) for _ in range(1)])
        self.transup2 = nn.Sequential(
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1),
        )
        self.res_block3 = nn.Sequential(*[ResidualInResidualDenseBlock(24) for _ in range(1)])
        self.main = nn.Sequential(
            nn.BatchNorm2d(24),
            nn.ReLU( inplace=True),
            nn.Conv2d(24, 12,kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(12),
            nn.ReLU( inplace=True),
            nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, z):
        out1 = self.conv1(z)
        out2 = self.res_block1(out1)  
        out3 = self.transup1(out2)    
        out4 = self.res_block2(out3)  
        out5 = self.transup2(out4)
        out6 = self.res_block3(out5) 
        img = self.main(out6)     
        return img

    def _n_parameters(self):
        n_params= 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params
