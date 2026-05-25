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
from rockpool.devices.xylo.syns65302 import XyloSamna, XyloSim
import samna
import pickle
import sys
import time

from rockpool.nn.modules import to_nir, LinearTorch, LIFTorch
from rockpool.nn.combinators import Sequential
import nir

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
ckpt_dir = Path(r"/home/danielmk/repos/xylo/scripts/checkpoints")

synnet_ckpts = sorted(
    p for p in ckpt_dir.iterdir()
    if p.is_file() and "sntc" in p.name
)

synnet_ckpts = sorted(
    synnet_ckpts,
    key=lambda p: torch.load(p, map_location="cpu").get("epoch", 0)
)

checkpoints = [
    torch.load(path, map_location="cpu")
    for path in synnet_ckpts
]

epoch = 5500

curr_ckpt = [x for x in checkpoints if x['epoch'] == epoch][0]

net = SynNet(
    n_channels=16,
    n_classes=1,
    size_hidden_layers=[140, 40, 40, 40, 40, 40],
    time_constants_per_layer=[2, 2, 4, 4, 8, 8],
    output="spikes",
    threshold=0.5,
    threshold_out=1.2,
    train_time_constants=True,
    # train_threshold=True,
)

net.load_state_dict(curr_ckpt["model_state"])

spec = mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')
sys.exit()
# quantizing the model
# spec.update(q.channel_quantize(**spec))
quantized_spec = q.global_quantize(**spec, bits_per_weight = 20)

spec.update(q.global_quantize(**spec, bits_per_weight = 20))

quantized_net = XyloSim.from_specification(**spec)

nir_graph = to_nir(net)

nir.write("sntc_epoch_5500.nir", nir_graph)


