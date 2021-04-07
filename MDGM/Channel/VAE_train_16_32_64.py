import argparse
import os
import numpy as np
import itertools
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch
import h5py
from load_data import load_data_3scales
from  CONV_VAE_model import Encoder, Decoder

parser = argparse.ArgumentParser()
parser.add_argument('--exp',type = str ,default = 'Channel_16_32_64',help = 'dataset')
parser.add_argument("--n-epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument('--n-train', type=int, default=40000, help='number of training data')
parser.add_argument('--n-test', type=int, default=200, help='number of test data')
parser.add_argument("--batch-size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample-interval", type=int, default=5, help="interval between image sampling")
parser.add_argument("--beta_vae", type=float, default=0.7, help="beta hyperparameter")
args = parser.parse_args()

dir = os.getcwd()
directory = f'/Channel/experiments/experiments_16_32_64/latent256/beta_0.7_{args.beta_vae}'
exp_dir = dir + directory + "/N{}_Bts{}_Eps{}_lr{}".\
    format(args.n_train, args.batch_size, args.n_epochs, args.lr)
output_dir = exp_dir + "/save_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
encoder = Encoder() 
decoder = Decoder(input_channels=3)
encoder.to(device)
decoder.to(device)
print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()))


train_hdf5_file = os.getcwd() + \
    f'/Channel/data/training_set_64-4_channel_40000.hdf5'
train_hdf5_file1 = os.getcwd() + \
    f'/Channel/data/training_set_32-4_channel_40000.hdf5'
train_hdf5_file2 = os.getcwd() + \
    f'/Channel/data/training_set_16-4_channel_40000.hdf5'

train_loader = load_data_3scales(train_hdf5_file,train_hdf5_file1,train_hdf5_file2, args.n_train, args.batch_size,singlescale=False)



optimizer= torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))



def load_model(size):
        dir = os.getcwd()
        if size == 16:
            model_dir = dir+f'/Channel/encoder_16_VAE_1_epoch30.pth'
            model = Encoder()
        elif size == 32:
            model_dir = dir+f'/Channel/encoder_16_32_VAE_1_0.7_epoch50.pth'
            model = Encoder()
        model.load_state_dict(torch.load(model_dir, map_location=device))
        encoder_model =model.to(device)
        return encoder_model

def X_l_encoder(encoder_model,xl_data):
        encoder_model_16.eval()
        z_l = encoder_model(xl_data)
        return z_l

def loss_function(recon_x, x, mu, logvar):
    rencon_loss = F.mse_loss(recon_x.view(-1,4096), x.view(-1,4096), size_average=False)
    mu=mu.reshape(-1,256)
    logvar=logvar.reshape(-1,256)
    KLD = torch.sum(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
    return rencon_loss + args.beta_vae*KLD, rencon_loss , KLD

class upsample(nn.Module):
    def __init__(self):
        super(upsample, self).__init__()
        self.transup = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
    def forward(self,zl,size):
        if size == 16:
            zl = self.transup(zl)
            zl = self.transup(zl)
        if size == 32:
           zl = self.transup(zl)
        return zl
# ----------#
#  Training #
# ----------#
upsample = upsample()
upsample.to(device)
encoder_model_16 = load_model(16) 
encoder_model_32 = load_model(32) 

for epoch in range(1,args.n_epochs+1):
    encoder.train()
    decoder.train()
    train_loss = 0   
    for batch_idx, (x64_data,x32_data,x16_data ) in enumerate(train_loader):
        x64_data = x64_data.to(device)
        x32_data = x32_data.to(device)
        x16_data = x16_data.to(device)
        optimizer.zero_grad() 
        z, mu, logvar = encoder(x64_data)
        _,z32,_ = X_l_encoder(encoder_model_32,x32_data)
        z32 = upsample(z32,32)
        _,z16,_ = X_l_encoder(encoder_model_16,x16_data)
        z16 = upsample(z16,16)
        z_h = torch.cat((z16,z32,z),1)
        recon_batch = decoder(z_h)
        loss,rec_loss, kl_loss= loss_function(recon_batch, x64_data,  mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} recon_loss:{:.6f} kl_loss:{:.6f}'.format(
                epoch, batch_idx * len(x64_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(x64_data),rec_loss.item() / len(x64_data),kl_loss.item() / len(x64_data)))
    batches_done = epoch * len(train_loader) + batch_idx
    if (epoch) % args.sample_interval == 0:
        torch.save(decoder.state_dict(), output_dir + f'/decoder_16_32_64_VAE_1_0.7_{args.beta_vae}_epoch{epoch}.pth')
        torch.save(encoder.state_dict(), output_dir + f'/encoder_16_32_64_VAE_1_0.7_{args.beta_vae}_epoch{epoch}.pth')
