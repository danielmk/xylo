#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:49:15 2026

@author: daniel-muller
"""

from rockpool.nn.networks import SynNet
import tables
from torch.optim import Adam, SGD
from torch.nn import MSELoss
import numpy as np
from rockpool.timeseries import TSEvent
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
import sys

torch.manual_seed(65)
np.random.seed(68)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

# dataset_path = r'/flash/FukaiU/danielmk/dataset.h5'
dataset_path = r'Y:\danielmk\okeon\dataset.h5'

dataset = tables.open_file(dataset_path, mode='r+')

net = SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [140, 40, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output='vmem',
    threshold=0.5,
    )

net = net.to(device)

print(net)

"""SELECT RANDOM SAMPLES"""
n = dataset.root.train.samples.shape[0]
k = 100           # number to select
t_stop=2.504

indices = np.random.choice(n, size=k, replace=False)

test_index = 18

sr=dataset.root.train.samples[test_index]['sr']
D = librosa.amplitude_to_db(np.abs(librosa.stft(dataset.root.train.audio[indices[test_index]])), ref=np.max)
fig, ax = plt.subplots(2, 1)
img = librosa.display.specshow(D, y_axis='linear', x_axis='s', sr=sr, ax=ax[0])

def to_raster(times : list, channels : list, t_start=0.0, t_stop=2.504, dt=0.001):
    
    if len(times) != len(channels): raise ValueError
    
    rasters = []
    for i in range(len(times)):
        event = TSEvent(
            times=times[i],
            channels=channels[i],
            t_stop=t_stop
            )
        raster = event.raster(dt, t_start=t_start, t_stop=t_stop, add_events=True)
        
        rasters.append(raster)
    return torch.Tensor(rasters)
    
rasters = to_raster(dataset.root.train.spike_times[indices], dataset.root.train.spike_channels[indices],t_stop=t_stop,dt=0.001)

sys.exit()

labels = np.zeros((k, int(t_stop / net.dt), 1))

onset_idx = int(1.0 / net.dt)

labels[:, onset_idx:onset_idx + int(0.5 / net.dt), :] = 1

labels = torch.Tensor(labels)

rasters, labels = rasters.to(device), labels.to(device)

# output, state, rec = net(torch.Tensor(rasters), record=True)

# - Get the optimiser functions
optimizer = Adam(net.parameters().astorch(), lr=1e-5)

# - Loss function
loss_fun = MSELoss().to(device=device)

net.train()

loss_t = []
for i in range(500):
    
    print(i)

    # events = events.to_dense()
    optimizer.zero_grad()
    
    output, _, _ = net(rasters, record=False)
    
    output = output.to(device)
    
    loss = loss_fun(output, labels)
    
    loss.backward()
    optimizer.step()

    this_loss = loss.item()
    
    loss_t.append(this_loss)

t_axis = np.arange(0,t_stop, net.dt)

