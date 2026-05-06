# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:00:25 2026

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import tables
from matplotlib.animation import FuncAnimation, FFMpegWriter
import soundfile as sf
from rockpool.nn.networks import SynNet
import torch
import xylo
from pathlib import Path


dataset_path = r'Y:\danielmk\okeon\dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

high_quality = np.argwhere(dst.root.train.quality_rating.read() == 3)[:, 0]

# example_idx = high_quality[2]
example_idx = 1

sr=44100

audio = dst.root.train.audio[example_idx]

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device('cpu')

test = dst.root.test

q = test.quality_rating[:]
species = test.samples.col("species")
species = np.array([s.decode() if isinstance(s, bytes) else s for s in species])

signal_idx = np.where(
    (q > 1) & (species != "None")
)[0]

noise_idx = np.where(
    species == "None"
)[0]

rng = np.random.default_rng()

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
    output="spikes",
    threshold=0.5,
    threshold_out=1.2,
    # train_time_constants=True,
    # train_threshold=True,

).to(device)

net.eval()

ckpt_dir = Path(r"C:\Users\Daniel\repos\xylo\scripts\checkpoints")

synnet_ckpts = sorted(
    p for p in ckpt_dir.iterdir()
    if p.is_file() and "synnet-long_" in p.name
)


synnet_ckpts = sorted(
    synnet_ckpts,
    key=lambda p: torch.load(p, map_location="cpu").get("epoch", 0)
)

checkpoints = [
    torch.load(path, map_location="cpu")
    for path in synnet_ckpts
]

epoch = 5000

curr_ckpt = [x for x in checkpoints if x['epoch'] == epoch][0]

net.load_state_dict(curr_ckpt["model_state"])
net.eval()                # important!

"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

print("Building rasters...")
all_rasters = xylo.training.build_all_rasters(test, t_stop, net.dt, net.size_in)

print("Building labels...")
all_labels = xylo.training.build_all_labels(test, species, t_stop, net.dt, net.size_out)

# Move **once**
all_rasters = all_rasters.to(device)
all_labels = all_labels.to(device)

output, out2, out3 = net(all_rasters, record=True)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 16})

t = np.arange(0, 2.5, 1/sr)

t_spikes = np.arange(0, t_stop, net.dt)

fig, ax = plt.subplots(
    3, 1,
    figsize=(16, 9),   # 16:9, safe
    dpi=100            # safe DPI
)

ax[0].plot(t, audio, color='k', linewidth=0.5)
ax[0].set_ylabel("Raw Audio")
ax[0].set_xticklabels([])

D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='s', sr=sr, ax=ax[1])
ax[1].set_xlabel("")
ax[1].set_xticklabels([])

spike_times = np.argwhere(output[example_idx, :, 0]) * net.dt
ax[2].plot(t_spikes, out3['out_neurons']['vmem'].detach()[example_idx, :, 0], color='k')
ax[2].vlines(spike_times, ymin=1.3, ymax=1.4, color='r')
# ax[2].vlines(spike_times, ymin=2.5, ymax=2.7, color='r')

ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Output Voltage")

# Exclude long calls
# precise_intervals = precise_intervals[(precise_intervals[:, 1] - precise_intervals[:, 0]) <= 2, :]
