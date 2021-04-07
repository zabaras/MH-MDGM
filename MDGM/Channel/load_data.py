import torch
from torch.utils.data import DataLoader, TensorDataset
from argparse import Namespace
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import h5py
import json
import os



def load_data_1scale(hdf5_file, ndata, batch_size, singlescale=True):
    with h5py.File(hdf5_file, 'r') as f:
        x_data = f['train'][:ndata]
    
    data_tuple = (torch.FloatTensor(x_data), ) if singlescale else (
            torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    data_loader = DataLoader(TensorDataset(*data_tuple),
        batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader


def load_data_2scales(hdf5_file,hdf5_file1, ndata, batch_size, singlescale=False):
    with h5py.File(hdf5_file, 'r') as f:
        x2_data = f['train'][:ndata]
        
    with h5py.File(hdf5_file1, 'r') as f:
        x1_data = f['train'][:ndata]
    
    data_tuple = (torch.FloatTensor(x_data), ) if singlescale else (
            torch.FloatTensor(x2_data), torch.FloatTensor(x1_data))
    data_loader = DataLoader(TensorDataset(*data_tuple),
        batch_size=batch_size, shuffle=True, drop_last=True)
    print(f'Loaded dataset: {hdf5_file}')
    return data_loader



def load_data_3scales(hdf5_file,hdf5_file1,hdf5_file2, ndata, batch_size, singlescale=False):
    with h5py.File(hdf5_file, 'r') as f:
        x3_data = f['train'][:ndata]
        
    with h5py.File(hdf5_file1, 'r') as f:
        x2_data = f['train'][:ndata]

    with h5py.File(hdf5_file2, 'r') as f:
        x1_data = f['train'][:ndata]

    data_tuple = (torch.FloatTensor(x_data), ) if singlescale else (
            torch.FloatTensor(x3_data), torch.FloatTensor(x2_data),torch.FloatTensor(x1_data))
    data_loader = DataLoader(TensorDataset(*data_tuple),
        batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader


