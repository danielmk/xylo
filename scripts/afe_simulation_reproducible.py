# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:00:25 2026

@author: Daniel
"""

import librosa
from rockpool.devices.xylo.syns65302 import AFESimExternal
import numpy as np
import matplotlib.pyplot as plt

# okeon_path = 'Y:\danielmk\okeon'

filepath = r'./scream_sample.wav'

test_sample, sr = librosa.load(filepath, sr=None)

dt_s = 0.009994

afesim_external = AFESimExternal.from_specification(
    spike_gen_mode="divisive_norm",
    fixed_threshold_vec=None,
    rate_scale_factor=63,
    low_pass_averaging_window=84e-3,
    dn_EPS=32,
    dt=dt_s,
)

# test_sample_int = np.round(test_sample * (2**31 - 1)).astype(int)

out_external,_,_ = afesim_external((test_sample, sr))

print(out_external)