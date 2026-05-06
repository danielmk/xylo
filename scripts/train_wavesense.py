# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:34:58 2026

@author: Daniel
"""

import tables
import numpy as np
from rockpool.nn.modules import LIFTorch  
from rockpool.nn.networks.wavesense import WaveSenseNet
from torch.optim import Adam, SGD
from torch.nn import BCEWithLogitsLoss
from rockpool.timeseries import TSEvent
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
import sys
import pdb
from rockpool.parameters import Constant


"""HYPERPARAMETERS"""
t_stop=2.504
batch_size=64

"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device("cpu")

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

dilations = [2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8]
n_out_neurons = 1
n_inp_neurons = 16
n_neurons = 16
kernel_size = 2
tau_mem = 0.002
base_tau_syn = 0.002
tau_lp = 0.01
threshold = 0.5
dt = 0.001


net = WaveSenseNet(
    dilations=dilations,
    n_classes=1,
    n_channels_in=16,
    n_channels_res=4,
    n_channels_skip=8,
    n_hidden=32,
    kernel_size=2,
    bias=Constant(0.0),
    smooth_output=True,
    tau_mem=Constant(0.002),
    base_tau_syn=0.002,
    tau_lp=tau_lp,
    threshold=Constant(threshold),
    neuron_model=LIFTorch,
    dt=dt,
).to(device)

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

def build_all_labels(train, species):
    n = train.samples.nrows

    labels = torch.zeros((n, net.size_out), dtype=torch.float32)

    for i, sample in enumerate(train.samples):
        if species[i] == "None":
            continue

        labels[i] = 1

    return labels

print("Building rasters...")
all_rasters = build_all_rasters(train, t_stop, net.dt)

print("Building labels...")
all_labels = build_all_labels(train, species)

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

loss_fun = BCEWithLogitsLoss().to(device=device)

net.train()

onset = int(1 / 0.001)

loss_t = []
for epoch in range(10000):
        
    batch_idc = sample_batch(batch_size)

    rasters, labels = load_batch(batch_idc)

    # events = events.to_dense()
    optimizer.zero_grad()
    
    _, _, output = net(rasters, record=True)
        
    output = output['readout_output'][:,onset:,:].to(device)
    
    output = torch.max(output, dim=1)[0]
    
    loss = loss_fun(output, labels)
    
    this_loss = loss.item()
    
    if epoch % 500 == 0:
        save_checkpoint(
            rf"C:\Users\Daniel\repos\xylo\scripts\checkpoints\wavesense_checkpoint_epoch_{epoch:04d}.pt",
            net,
            optimizer,
            epoch,
            this_loss,
        )
    
    loss.backward()
    optimizer.step()



    loss_t.append(this_loss)

    print(epoch, this_loss)

