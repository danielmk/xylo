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
from rockpool.devices.xylo.syns65302 import XyloSim
import samna
import pickle
import sys
import time
from rockpool.timeseries import TSEvent

from rockpool.nn.modules import to_nir, LinearTorch, LIFTorch
from rockpool.nn.combinators import Sequential
import nir
import copy
from xylo import evaluation
import pdb

dataset_path = r'dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

test = dst.root.test

q_test = test.quality_rating[:]
species_test = test.samples.col("species")
species_test = np.array([s.decode() if isinstance(s, bytes) else s for s in species_test])

y_true_test = np.zeros((species_test.shape[0]))
y_true_test[species_test=='Ruddy Kingfisher'] = 1

# example_idx = high_quality[2]
example_idx = 1

sr=44100

test = dst.root.test

q_test = test.quality_rating[:]
species_test = test.samples.col("species")
species_test = np.array([s.decode() if isinstance(s, bytes) else s for s in species_test])

y_true_test = np.zeros((species_test.shape[0]))
y_true_test[species_test=='Ruddy Kingfisher'] = 1

rng = np.random.default_rng()

"""HYPERPARAMETERS"""
t_stop=2.504
batch_size=64

# ---------------------------------------------------------------------
# MODEL + OPTIMIZER (must match original training!)
# ---------------------------------------------------------------------
ckpt_dir = Path(r"/home/danielmk/repos/xylo/scripts/checkpoints")

synnet_ckpts = sorted(
    p for p in ckpt_dir.iterdir()
    if p.is_file() and "synnetv2" in p.name
)

synnet_ckpts = sorted(
    synnet_ckpts,
    key=lambda p: torch.load(p, map_location="cpu").get("epoch", 0)
)

checkpoints = [
    torch.load(path, map_location="cpu")
    for path in synnet_ckpts
]

epoch = 4000

curr_ckpt = [x for x in checkpoints if x['epoch'] == epoch][0]

net = SynNet(
    n_channels=16,
    n_classes=1,
    size_hidden_layers=[128, 64, 40, 40, 40, 40],
    time_constants_per_layer=[2, 2, 4, 4, 8, 8],
    output="spikes",
    threshold=0.5,
    threshold_out=1.8,
    #train_time_constants=True,
    #train_threshold=True,
)

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


all_rasters_test = build_all_rasters(test, t_stop, net.dt)

# all_rasters_test = np.array(all_rasters_test)

net.load_state_dict(curr_ckpt["model_state"])

spec = mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')

spec_pre = copy.copy(spec)

spec.update(q.global_quantize(**spec))

fig, ax = plt.subplots(3,1)
fig.suptitle("Pre Quantization")
ax[0].hist(spec_pre['weights_in'].flatten(), bins=100)
ax[1].hist(spec_pre['weights_rec'].flatten()[spec_pre['weights_rec'].flatten() !=0], bins=2**8)
ax[2].hist(spec_pre['weights_out'].flatten(), bins=100)
# quantizing the model

# quantized_spec = q.global_quantize(**spec, bits_per_weight = 20)

fig, ax = plt.subplots(3,1)
fig.suptitle("Post Quantization")
ax[0].hist(spec['weights_in'].flatten(), bins=100)
ax[1].hist(spec['weights_rec'].flatten()[spec['weights_rec'].flatten() !=0], bins=2**8)
ax[2].hist(spec['weights_out'].flatten(), bins=100)

"""
The recurrent weight matrix is nonzero in the top-right quarter, reflecting
that the first 63 neurons connect to the next 63 neurons (filled with the 
weights on the second linear layer of the model). Note that there are no 
recurrent connections as the diagonal blocks are all zero.
"""

quantized_net = XyloSim.from_specification(**spec)


output_test = []
for idx, curr_raster in enumerate(all_rasters_test):
    print(f"Current raster: {idx}")
    curr_output, _, _ = net(curr_raster, record=False)
    output_test.append(curr_output)

output_test = np.array(output_test)

pdb.set_trace()

y_pred_train = predict_events(net, all_rasters_train)
y_pred_test  = predict_events(net, all_rasters_test)

train_rates = evaluation.confusion_rates(y_true_train, y_pred_train)
test_rates = evaluation.confusion_rates(y_true_test, y_pred_test)

# nir_graph = to_nir(net)

# nir.write("sntc_epoch_5500.nir", nir_graph)


