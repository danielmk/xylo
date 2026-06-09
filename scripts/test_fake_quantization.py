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
from xylo import evaluation, nets
import pdb
from rockpool.nn.modules import LinearTorchQAT, LIFTorchQAT

net = nets.synnetqatv1('vmem')

spec = mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')

spec_pre = copy.copy(spec)

spec.update(q.global_quantize(**spec))

W_in_hw = spec["weights_in"]
W_rec_hw = spec["weights_rec"]
W_out_hw = spec["weights_out"]

th_hw = spec["threshold"]
th_out_hw = spec["threshold_out"]


linear_modules = [
    m for m in net.seq.modules()
    if isinstance(m, LinearTorchQAT)
]

input_module = linear_modules[0]
hidden_modules = linear_modules[1:-1]
output_module = linear_modules[-1]

global_scale = net.compute_global_scale(linear_modules)
output_scale = net.compute_output_scale(linear_modules)

def torch_to_int(W, scale):
    return torch.round(W * scale).clamp(-127, 127).to(torch.int32)

W_in_qat = torch_to_int(input_module.weight, global_scale)

W_rec_qat = [
    torch_to_int(m.weight, global_scale)
    for m in hidden_modules
]

W_out_qat = torch_to_int(output_module.weight, output_scale)

print("Input mismatch:",
      (W_in_qat.cpu().numpy() != W_in_hw).sum())

for i, (w_qat, w_hw) in enumerate(zip(W_rec_qat, W_rec_hw)):
    diff = (w_qat.cpu().numpy() != w_hw).sum()
    print(f"Hidden {i} mismatch:", diff)
    
print("Output mismatch:",
  (W_out_qat.cpu().numpy() != W_out_hw).sum())