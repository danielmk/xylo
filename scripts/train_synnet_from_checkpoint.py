# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:34:58 2026

@author: Daniel
"""
import os
import re
import torch
from pathlib import Path
import tables
import numpy as np
from rockpool.nn.networks import SynNet
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from rockpool.timeseries import TSEvent
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
import sys
import pdb

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

"""HYPERPARAMETERS"""
t_stop=2.504
batch_size=64

# ---------------------------------------------------------------------
# MODEL + OPTIMIZER (must match original training!)
# ---------------------------------------------------------------------

net = SynNet(
    n_channels=16,
    n_classes=1,
    size_hidden_layers=[140, 40, 40, 40, 40, 40],
    time_constants_per_layer=[2, 2, 4, 4, 8, 8],
    output="vmem",
    threshold=0.5,
    train_time_constants=True,
    train_threshold=True,
).to(device)


"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

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


# ---------------------------------------------------------------------
# CHECKPOINT UTILITIES
# ---------------------------------------------------------------------

CHECKPOINT_DIR = Path(r"C:\Users\Daniel\repos\xylo\scripts\checkpoints")

def find_latest_checkpoint(checkpoint_dir, prefix="sntcth_checkpoint_epoch"):
    """
    Find the checkpoint with the largest epoch number.
    """
    pattern = re.compile(rf"{prefix}_(\d+)\.pt")

    best_epoch = -1
    best_ckpt = None

    for p in checkpoint_dir.glob(f"{prefix}_*.pt"):
        m = pattern.search(p.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_ckpt = p

    return best_ckpt, best_epoch

optimizer = Adam(net.parameters().astorch(), lr=1e-5)
loss_fun = MSELoss().to(device)

# ---------------------------------------------------------------------
# LOAD LATEST CHECKPOINT (if any)
# ---------------------------------------------------------------------

ckpt_path, last_epoch = find_latest_checkpoint(CHECKPOINT_DIR)

start_epoch = 0

if ckpt_path is not None:
    print(f"Resuming from checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    net.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1

    print("Successfully restored model and optimizer")
    print(f"Resuming at epoch {start_epoch}")
else:
    print("No checkpoint found — starting from scratch")
    
net.train()

for epoch in range(start_epoch, start_epoch + 8501):

    batch_idc = sample_batch(batch_size)
    rasters, labels = load_batch(batch_idc)

    optimizer.zero_grad()

    output, _, _ = net(rasters, record=False)
    
    output=output.to(device)
    
    loss = loss_fun(output, labels)

    this_loss = loss.item()

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        save_checkpoint(
            CHECKPOINT_DIR / f"sntcth_checkpoint_epoch_{epoch:04d}.pt",
            net,
            optimizer,
            epoch,
            this_loss,
        )

    print(epoch, this_loss)