# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:00:25 2026

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import sys
import uuid
import pdb
import shelve
import pathlib
import librosa
import xylo
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import uniform_filter1d
import rockpool
from rockpool.devices.xylo.syns65302 import AFESimExternal
from rockpool.devices.xylo.syns65302 import AFESimPDM



datapath = xylo.Config.okeon_bucket

detections = pd.read_csv(datapath / "2sp_detect_forDaniel.csv")

filename = "CHATANOP_20200508_063000.flac"

location, date, number = filename.split('_')

filepath = datapath / location / date[:4] / filename

data, sr = librosa.load(filepath, duration=60, sr=None)

D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)

curr_detections = detections[(detections['Filename'] == filename) & (detections['Confidence'] >= 0.9)]

merged_intervals = xylo.features.merge_intervals_pandas(curr_detections)

"""
for row in merged_intervals.iloc:
    print(row['Time_start'], row['Time_End'])
    ax.fill_betweenx([0, 11025.0], x1=row['Time_start'], x2=row['Time_End'], color='r', alpha=0.25)
"""

# Target band
f_low = 1200.0   # Hz
f_high = 2500.0  # Hz

# Envelope / framing parameters
smooth_ms = 100           # smoothing window for envelope (ms)
frame_hop_ms = 10        # used for time axis spacing (ms); not required by librosa here

# Event logic
threshold_k = 3.0        # median + k * MAD (robust); increase if too sensitive
min_event_dur_ms = 60    # minimum duration of a valid event (ms)
min_silence_ms = 50      # minimum gap to split events (ms)

y_bp = xylo.features.bandpass_filter(data, sr, f_low, f_high, order=4)

analytic = hilbert(y_bp)

amp_env = np.abs(analytic)

smooth_samples = max(1, int(smooth_ms * 1e-3 * sr))
env_smooth = uniform_filter1d(amp_env, size=smooth_samples, mode="nearest")

thr = xylo.features.robust_threshold(env_smooth, k=threshold_k)

hop_sec = 1.0 / sr
min_event_dur = min_event_dur_ms * 1e-3
min_silence = min_silence_ms * 1e-3

precise_intervals = []
for r in merged_intervals.iloc:
    start_idx = r['Time_start'] * sr
    end_idx = r['Time_End'] * sr
    curr_interval = env_smooth[start_idx: end_idx]
    # on_idx, off_idx = xylo.features.detect_regions(
    #     curr_interval, sr, hop_sec, thr, min_event_dur, min_silence
    #     )
    on_idx, off_idx = xylo.features.detect_regions_single(
        curr_interval, thr)

    start_time = (on_idx + start_idx) / sr
    end_time = (off_idx + start_idx) / sr
    # if len(start_time) == 1 & len(end_time) == 1:
    precise_intervals.append([start_time, end_time])

precise_intervals = np.array(precise_intervals)

fig, ax = plt.subplots(2, 1)
img = librosa.display.specshow(D, y_axis='linear', x_axis='s', sr=sr, ax=ax[0])
t = np.arange(0, env_smooth.shape[0] * (1/sr), 1/sr)
ax[1].plot(t, env_smooth)

ax[0].set_xlim((0, 10))
ax[1].set_xlim((0, 10))

# Exclude long calls
precise_intervals = precise_intervals[(precise_intervals[:, 1] - precise_intervals[:, 0]) <= 2, :]

plt.figure()
img = librosa.display.specshow(D, y_axis='linear', x_axis='s')
plt.vlines(precise_intervals[:, 0], ymin=0, ymax=10000, color='r', alpha=0.8)
plt.vlines(precise_intervals[:, 1], ymin=0, ymax=10000, color='b', alpha=0.8)

dt_s = 0.009994

afesim_external = AFESimExternal.from_specification(spike_gen_mode="direct_threshold",
                                                    fixed_threshold_vec = None,
                                                    rate_scale_factor=63,
                                                    low_pass_averaging_window=0.0001,
                                                    dn_EPS=32,
                                                    dt=dt_s,
                                                    exact_hw=True
                                                   )

afesim_external._check_hw_types = False




audio = data.astype(np.float32)

audio /= np.max(np.abs(audio)) + 1e-9

audio_int = (audio * 32767).astype(np.int64)

out_external,_,_ = afesim_external((audio_int, sr))

#librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
#                         x_axis='time')

