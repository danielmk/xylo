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
from rockpool.devices.xylo.syns65302 import config_from_specification, mapper
import rockpool.transform.quantize_methods as q
from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as hdu
from rockpool.devices.xylo.syns65302 import XyloSamna
import samna


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

quality = test.quality_rating[:]
species = test.samples.col("species")
species = np.array([s.decode() if isinstance(s, bytes) else s for s in species])

signal_idx = np.where(
    (quality > 1) & (species != "None")
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

"""QUANTIZE AND BULID XYLO 3 CONFIGURATION"""
# getting the model specifications using the mapper function
spec = mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')
# quantizing the model
spec.update(q.channel_quantize(**spec))

xylo_conf, is_valid, msg = config_from_specification(**spec)

# Getting the connected devices and choosing XyloAudio 3 board
xylo_nodes = hdu.find_xylo_a3_boards()

if len(xylo_nodes) == 0:
    raise ValueError('A connected XyloAudio 3 development board is required for this tutorial.')

xa3 = xylo_nodes[0]

# Instantiating XyloSamna and deploying to the dev kit; make sure your dt corresponds to the dt of your input data
Xmod = XyloSamna(device=xa3, config=xylo_conf, dt = net.dt)

out, _, rec = Xmod(all_rasters, record=True)

# output, out2, out3 = net(all_rasters, record=True)




