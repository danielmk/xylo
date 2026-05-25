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
import pickle
import sys
import time
import pdb

dataset_path = r'dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

high_quality = np.argwhere(dst.root.test.quality_rating.read() == 3)[:, 0]

mid_quality = np.argwhere(dst.root.test.quality_rating.read() == 2)[:, 0]

noise_quality = np.argwhere(np.isnan(dst.root.test.quality_rating.read()))[:, 0]

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

net = SynNet(
    n_channels=16,
    n_classes=1,
    size_hidden_layers=[128, 64, 40, 40, 40, 40],
    time_constants_per_layer=[2, 2, 4, 4, 8, 8],
    output="spikes",
    threshold=0.5,
    threshold_out=1.2,
    # train_time_constants=True,
    # train_threshold=True,
)

ckpt_dir = Path(r"/home/danielmk/repos/xylo/scripts/checkpoints")

synnet_ckpts = sorted(
    p for p in ckpt_dir.iterdir()
    if p.is_file() and "synnetv2_" in p.name
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

net.load_state_dict(curr_ckpt["model_state"])

"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

print("Building rasters...")
all_rasters = xylo.training.build_all_rasters(test, t_stop, net.dt, net.size_in)

# Move **once**
all_rasters = all_rasters

thresholds = np.arange(1.8, 1.85, 0.1)

quantized_thresholds = []

all_outputs = []
all_states = []
all_recs = []

for th in thresholds:
    print(f"Threshold: {th}")
    net = SynNet(
        n_channels=16,
        n_classes=1,
        size_hidden_layers=[128, 64, 40, 40, 40, 40],
        time_constants_per_layer=[2, 2, 4, 4, 8, 8],
        output="spikes",
        threshold=0.5,
        threshold_out=th,
        # train_time_constants=True,
        # train_threshold=True,
    )
    
    # sys.exit()
    
    net.load_state_dict(curr_ckpt["model_state"])
    
    """QUANTIZE AND BULID XYLO 3 CONFIGURATION"""
    # getting the model specifications using the mapper function
    spec = mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')

    # quantizing the model
    # spec.update(q.channel_quantize(**spec))
    spec.update(q.global_quantize(**spec))
    
    quantized_threshold = spec['threshold_out']
    
    print(f"Threshold quantized: {quantized_threshold}")
    
    quantized_thresholds.append(quantized_threshold)
    
    xylo_conf, is_valid, msg = config_from_specification(**spec)
    
    # Getting the connected devices and choosing XyloAudio 3 board
    xylo_nodes = hdu.find_xylo_a3_boards()
    
    if len(xylo_nodes) == 0:
        raise ValueError('A connected XyloAudio 3 development board is required for this tutorial.')
    
    xa3 = xylo_nodes[0]
    
    # Instantiating XyloSamna and deploying to the dev kit; make sure your dt corresponds to the dt of your input data
    Xmod = XyloSamna(device=xa3, config=xylo_conf, dt=net.dt)
    
    time.sleep(5)

    out_list = []
    state_list = []
    rec_list = []
    
    for idx, raster in enumerate(all_rasters):
        print(f"Curr idx: {idx}")
        out, state, rec = Xmod(raster, record=False, record_power=True)
        out_list.append(out)
        state_list.append(state)
        rec_list.append(rec)
    
    all_outputs.append(out_list)
    all_states.append(state_list)
    all_recs.append(rec_list)
    

np.savez(f'synnetv2_{epoch}_accelerate_time_xylo_spikes_with_power.npz',
         xylo_output=all_outputs,
         thresholds=thresholds,
         quantized_thresholds=quantized_thresholds,
         states=all_states,
         recs=all_recs)
    
    # output, out2, out3 = net(all_rasters, record=True)




