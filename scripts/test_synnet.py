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
from rockpool.timeseries import TSEvent
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
import sys
import pdb
from pathlib import Path


"""HYPERPARAMETERS"""
t_stop=2.504
batch_size=64

"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

# dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device("cpu")

dataset_path = r'Y:\danielmk\okeon\dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

test = dst.root.test

q = test.quality_rating[:]
species_test = test.samples.col("species")
species_test = np.array([s.decode() if isinstance(s, bytes) else s for s in species_test])

y_true_test = np.zeros((species_test.shape[0]))
y_true_test[species_test=='Ruddy Kingfisher'] = 1

train = dst.root.train

q = train.quality_rating[:]
species_train = train.samples.col("species")
species_train = np.array([s.decode() if isinstance(s, bytes) else s for s in species_train])

y_true_train = np.zeros((species_train.shape[0]))
y_true_train[species_train=='Ruddy Kingfisher'] = 1

rng = np.random.default_rng()

net = SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [140, 40, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output='spikes',
    threshold=0.5,
    threshold_out=1.5
    )

net = net.to(device)

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
all_rasters_test = build_all_rasters(test, t_stop, net.dt)

all_rasters_train = build_all_rasters(train, t_stop, net.dt)

# Move **once**
all_rasters_test = all_rasters_test.to(device)

all_rasters_train = all_rasters_train.to(device)

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

def balanced_accuracy(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # recall
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # specificity

    return 0.5 * (tpr + tnr)

ckpt_dir = Path(r"C:\Users\Daniel\repos\xylo\scripts\checkpoints")

synnet_ckpts = sorted(
    p for p in ckpt_dir.iterdir()
    if p.is_file() and "synnet" in p.name
)


synnet_ckpts = sorted(
    synnet_ckpts,
    key=lambda p: torch.load(p, map_location="cpu").get("epoch", 0)
)

checkpoints = [
    torch.load(path, map_location="cpu")
    for path in synnet_ckpts
]


threshold_grid = np.arange(1.0, 2.0, 0.1)
balanced_training_accuracies = []
balanced_test_accuracies = []
for ckpt in checkpoints:
    curr_baltrainacc = []
    curr_baltestacc = []
    for thr in threshold_grid:

        net = SynNet(
            n_channels = 16,
            n_classes = 1,
            size_hidden_layers = [140, 40, 40, 40, 40, 40],
            time_constants_per_layer = [2, 2, 4, 4, 8, 8],
            output='spikes',
            threshold=0.5,
            threshold_out=thr
            )

        net = net.to(device)
        
        net.load_state_dict(ckpt["model_state"])
        net.eval()                # important!
        
        output_train, _, _ = net(all_rasters_train, record=False)
        output_test, _, _ = net(all_rasters_test, record=False)
        
        y_pred_test = torch.any(output_test[:, int(1.0 / net.dt):, 0] == 1, axis=1)
        y_pred_train = torch.any(output_train[:, int(1.0 / net.dt):, 0] == 1, axis=1)
        
        ba_test = balanced_accuracy(y_true_test, np.array(y_pred_test))
        ba_train = balanced_accuracy(y_true_train, np.array(y_pred_train))
        
        curr_baltrainacc.append(ba_train)
        curr_baltestacc.append(ba_test)
        print(f"Threshold: {thr}, Epoch: {ckpt['epoch']}, Train BA: {ba_train}, Test BA: {ba_test}")

    balanced_training_accuracies.append(curr_baltrainacc)
    balanced_test_accuracies.append(curr_baltestacc)


ckpt = torch.load(
    r"C:\Users\Daniel\repos\xylo\scripts\checkpoints\synnet_checkpoint_epoch_0950.pt",
    map_location="cpu")   # safe, works even if it was trained on GPU


net.load_state_dict(ckpt["model_state"])
net.eval()                # important!


output_train, _, _ = net(all_rasters_train, record=False)

output_test, _, _ = net(all_rasters_test, record=False)

y_pred_test = torch.any(output_test[:, int(1.0 / net.dt):, 0] == 1, axis=1)

y_pred_train = torch.any(output_train[:, int(1.0 / net.dt):, 0] == 1, axis=1)

ba_test = balanced_accuracy(y_true_test, np.array(y_pred_test))

ba_train = balanced_accuracy(y_true_train, np.array(y_pred_train))
