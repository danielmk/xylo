# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:34:58 2026

@author: Daniel
"""

import tables
import numpy as np
from rockpool.nn.networks import SynNet
from torch.optim import Adam, SGD
from torch.nn import MSELoss
import torch.nn as nn
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.timeseries import TSEvent
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
import sys
import pdb
from rockpool.nn.combinators import Sequential

"""HYPERPARAMETERS"""
t_stop=2.504
batch_size=64

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

rng = np.random.default_rng()

net = Sequential(LinearTorch((16, 512)),
                LIFTorch((512), has_rec=True),
                LinearTorch((512, 64)),
                LIFTorch((64)),
                LinearTorch((64, 64)),
                LIFTorch((64)),
                LinearTorch((64,1)),
                LIFTorch((1)))

net = net.to(device)

sys.exit()

def build_all_rasters(train, t_stop, dt):
    n = train.spike_times.nrows
    n_steps = int(t_stop / dt)

    rasters_np = np.zeros((n, n_steps, net.size_in), dtype=np.float32)

    for i in range(n):
        event = TSEvent(
            times=train.spike_times[i],
            channels=train.spike_channels[i],
            t_stop=t_stop
        )
        rasters_np[i] = event.raster(
            dt, t_start=0.0, t_stop=t_stop, add_events=True
        )

    return torch.from_numpy(rasters_np)

def build_all_labels(train, species, t_stop, dt, label_amplitude=1.0):
    n = train.samples.nrows
    n_steps = int(t_stop / dt)

    labels = torch.zeros((n, n_steps, net.size_out), dtype=torch.float32)

    for i, sample in enumerate(train.samples):
        if species[i] == "None":
            continue

        start = int(1 / dt)
        stop = start + int(sample["call_duration"] / dt)
        labels[i, start:stop, 0] = label_amplitude

    return labels

print("Building rasters...")
all_rasters = build_all_rasters(train, t_stop, net.dt)

print("Building labels...")
all_labels = build_all_labels(train, species, t_stop, net.dt)

# Move **once**
all_rasters = all_rasters.to(device)
all_labels = all_labels.to(device)

def sample_batch(batch_size):
    half = batch_size // 2

    sig = rng.choice(signal_idx, size=half, replace=False)
    noi = rng.choice(noise_idx, size=half, replace=False)

    idx = np.concatenate([sig, noi])
    rng.shuffle(idx)

    return torch.as_tensor(idx, device=device)

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



def load_batch(batch_idx):
    return (
        all_rasters[batch_idx],
        all_labels[batch_idx]
    )

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    loss,
    extra=None
):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }

    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)

optimizer = Adam(net.parameters().astorch(), lr=1e-5)

loss_fun = MSELoss().to(device=device)

net.train()

loss_t = []
for epoch in range(1000):
        
    batch_idc = sample_batch(batch_size)

    rasters, labels = load_batch(batch_idc)

    # events = events.to_dense()
    optimizer.zero_grad()
    
    output, _, _ = net(rasters, record=False)
    
    output = output.to(device)
    
    loss = loss_fun(output, labels)
    
    this_loss = loss.item()
    
    if epoch % 50 == 0:
        save_checkpoint(
            rf"C:\Users\Daniel\repos\xylo\scripts\checkpoints\recnet_checkpoint_epoch_{epoch:04d}.pt",
            net,
            optimizer,
            epoch,
            this_loss,
        )
    
    loss.backward()
    optimizer.step()



    loss_t.append(this_loss)

    print(epoch, this_loss)

