# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:34:58 2026

@author: Daniel
"""

import tables
import numpy as np
from rockpool.nn.networks import SynNet
from torch.optim import AdamW, SGD, Adam
from torch.nn import MSELoss
from rockpool.timeseries import TSEvent
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
import sys
import pdb
import xylo

"""HYPERPARAMETERS"""
t_stop = 2.504
batch_size = 256

"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

dataset_path = r'Y:\danielmk\okeon\dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

train = dst.root.train

q = train.quality_rating[:]
species = train.samples.col("species")
species = np.array([s.decode() if isinstance(s, bytes) else s for s in species])

signal_idx = np.where(
    (q > 1) & (species != "None")
)[0]

noise_idx = np.where(
    species == "None"
)[0]

net = xylo.nets.synnetv4(output='vmem').to(device)

print("Building rasters...")
all_rasters = xylo.training.build_all_rasters(train, t_stop, net.dt, net.size_in).to(device)

print("Building labels...")
all_labels = xylo.training.build_all_labels(train, species, t_stop, net.dt, net.size_out, label_amplitude=1.0).to(device)

print("Labels Done.")

optimizer = AdamW(net.parameters().astorch(), lr=1e-5, weight_decay=1e-4)

loss_fun = MSELoss().to(device=device)

net.train()

loss_t = []
for epoch in range(10000):
        
    batch_idc = xylo.training.sample_batch(batch_size, signal_idx, noise_idx)

    # rasters, labels = load_batch(batch_idc)
    rasters, labels = all_rasters[batch_idc], all_labels[batch_idc]

    # events = events.to_dense()
    optimizer.zero_grad()
    
    output, _, rec = net(rasters, record=False)
    
    output = output.to(device)
    
    loss = loss_fun(output, labels)
    
    this_loss = loss.item()
    
    if epoch % 50 == 0:
        xylo.training.save_checkpoint(
            rf"C:\Users\Daniel\repos\xylo\scripts\checkpoints\synnetv4_checkpoint_epoch_{epoch:04d}.pt",
            net,
            optimizer,
            epoch,
            this_loss,
        )
    
    loss.backward()
    optimizer.step()

    loss_t.append(this_loss)

    print(epoch, this_loss)

