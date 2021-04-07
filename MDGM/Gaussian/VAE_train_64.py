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
from load_data import load_data_1scale
from  CONV_VAE_model import Encoder, Decoder

parser = argparse.ArgumentParser()
parser.add_argument("--n-epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument('--n-train', type=int, default=25000, help='number of training data')
parser.add_argument('--n-test', type=int, default=1000, help='number of test data')
parser.add_argument("--batch-size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample-interval", type=int, default=5, help="interval between image sampling")
parser.add_argument("--beta_vae", type=float, default=0.5, help="beta hyperparameter")
args = parser.parse_args()

dir = os.getcwd()
directory = f'/Gaussian/experiments/experiments_64/latent256/beta_{args.beta_vae}'
exp_dir = dir + directory + "/N{}_Bts{}_Eps{}_lr{}".\
    format(args.n_train, args.batch_size, args.n_epochs, args.lr)
output_dir = exp_dir + "/save_model"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
encoder = Encoder() 
decoder = Decoder()
encoder.to(device)
decoder.to(device)
print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()))


train_hdf5_file = os.getcwd() + \
    f'/Gaussian/data/training_set_64_gaussian1_25000.hdf5'
train_loader = load_data_1scale(train_hdf5_file, args.n_train, args.batch_size,singlescale=True)


optimizer= torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))


def loss_function(recon_x, x, mu, logvar):
    Recon_loss = F.mse_loss(recon_x.view(-1,4096), x.view(-1,4096), size_average=False)
    mu=mu.reshape(-1,256)
    logvar=logvar.reshape(-1,256)
    KLD = torch.sum(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
    return Recon_loss + args.beta_vae*KLD, Recon_loss , KLD
# ----------#
#  Training #
# ----------#
for epoch in range(1,args.n_epochs+1):
    encoder.train()
    decoder.train()
    train_loss = 0
    for batch_idx, (data, ) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        z, mu, logvar = encoder(data)
        recon_batch = decoder(z)
        loss,rec_loss, kl_loss= loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} recon_loss:{:.6f} kl_loss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data),rec_loss.item() / len(data),kl_loss.item() / len(data)))

    batches_done = epoch * len(train_loader) + batch_idx
    if (epoch) % args.sample_interval == 0:
        torch.save(decoder.state_dict(), output_dir + f'/decoder_64_VAE_{args.beta_vae}_epoch{epoch}.pth')
        torch.save(encoder.state_dict(), output_dir + f'/encoder_64_VAE_{args.beta_vae}_epoch{epoch}.pth')

