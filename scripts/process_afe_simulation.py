# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:00:25 2026

@author: Daniel
"""

import numpy as np
import pandas as pd
import librosa
import xylo
from rockpool.devices.xylo.syns65302 import AFESimExternal
import matplotlib.pyplot as plt
import matplotlib as mpl
from rockpool.timeseries import (
    TimeSeries,
    TSContinuous,
    TSEvent,
    set_global_ts_plotting_backend,
)

datapath = xylo.Config.okeon_bucket

detections = pd.read_csv(datapath / "2sp_detect_forDaniel.csv")

filename = "CHATANOP_20200508_063000.flac"

location, date, number = filename.split('_')

filepath = datapath / location / date[:4] / filename

data, sr = librosa.load(filepath, duration=10, sr=None)

dt_s = 0.009994
# dt_s = 0.000102

afesim_external = AFESimExternal.from_specification(
    spike_gen_mode="divisive_norm",
    fixed_threshold_vec=None,
    dt=dt_s,
)

out_external, _, _ = afesim_external((data, sr))

plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["font.size"] = 18

fig, ax = plt.subplots(2, 1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='s', sr=sr, ax=ax[0])

img_spikes = ax[1].imshow(out_external.T, aspect='auto', extent=[0, 10,1,16], origin='lower', interpolation='None')
ax[1].set_ylabel("Input Neuron")
fig.colorbar(img, ax=ax[0], format="%+2.f dB")
fig.colorbar(img_spikes, ax=ax[1], format="%2.f Spikes")
# ax[0].set_xlim((0, 10))
# ax[1].set_ylim((0, 10))

spikes = TSEvent.from_raster(out_external, dt=dt_s)

fig, ax = plt.subplots(2, 1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='s', sr=sr, ax=ax[0])

img_spikes = ax[1].scatter(spikes.times,spikes.channels, marker='|', alpha=0.5)
ax[1].set_ylabel("Input Neuron")
ax[1].set_xlim((0, 10))
fig.colorbar(img, ax=ax[0], format="%+2.f dB")
fig.colorbar(img_spikes, ax=ax[1], format="%2.f Spikes")
