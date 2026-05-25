# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:30:57 2026

@author: Daniel
"""

import xylo

import os
import sys
import pdb
import tables
import pandas as pd
import numpy as np
import librosa
import pdb
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from rockpool.devices.xylo.syns65302 import AFESimExternal
from rockpool.timeseries import TSEvent

CONFIDENCE_THRESHOLD = 0.0
SPECIES = 'Okinawa Rail'

# SPECIES TARGET BAND
F_LOW = 300.0
F_HIGH = 5000.0
BANDPASS_ORDER = 4

# ENVELOPE / FRAMING PARAMETERS
SMOOTH_MS = 100
FRAME_HOP_MS = 10

# EVENT LOGIC
THRESHOLD_K = 3.0
MIN_EVENT_DUR = 1.0
MIN_SILENCE = 1.0

# EXTRACTION TIMES
TIME_PRE = 1.0  # in seconds
TIME_POST = 1.5
target_dur = TIME_PRE + TIME_POST

# dataset_path = r'/flash/FukaiU/danielmk/dataset.h5'
dataset_path = r'/flash/FukaiU/danielmk/dataset.h5'


dataset = tables.open_file(dataset_path, mode='r+')

detections = pd.read_csv(xylo.Config.okeon_bucket / "2sp_detect_forDaniel.csv")

all_files = detections[(detections['Confidence'] >= CONFIDENCE_THRESHOLD) & (detections['Species_Name'] == SPECIES)]['Filename'].unique()

samples = dataset.root.train.samples.read()

processed_files = samples[(samples['species'] == 'None')]['filename']

processed_files = np.char.decode(processed_files, encoding="utf-8", errors="strict")

mask = ~np.isin(all_files, processed_files)

# Use the mask to filter array_a
unprocessed_files = all_files[mask]

for curr_file in unprocessed_files:

    location, date, number = curr_file.split('_')
    
    audio, sr = librosa.load(xylo.Config.okeon_bucket / location / date[:4] / curr_file, duration=60, sr=None)
    
    target_dur_idx = target_dur * sr
            
    filtered_audio = xylo.features.bandpass_filter(audio, sr, F_LOW, F_HIGH, order=BANDPASS_ORDER)

    analytic_audio = hilbert(filtered_audio)

    env_audio = np.abs(analytic_audio)

    smooth_samples = max(1, int(SMOOTH_MS * 1e-3 * sr))
    env_smooth = uniform_filter1d(env_audio, size=smooth_samples, mode="nearest")

    thr = xylo.features.robust_threshold(env_smooth, k=THRESHOLD_K)
        
    on_idx, off_idx = xylo.features.detect_regions_single(env_smooth, thr)
    
    on_idx = np.array(on_idx)

    # on_idx = on_idx[on_idx > int(TIME_PRE * sr)]
    
    if not np.any(on_idx) or not np.any(off_idx):
        continue
    
    if on_idx <= TIME_PRE * sr:
        continue
    
    if on_idx >= (60 * sr - TIME_POST * sr):
        continue

    call_dur = (off_idx - on_idx) / sr
    
    precise_start = on_idx - int(TIME_PRE * sr)
    
    precise_stop = on_idx + int(TIME_POST * sr)
    
    precise_interval_audio = audio[precise_start:precise_stop]
    
    t = np.arange(0, precise_interval_audio.shape[0] / sr, 1 /sr)

    dt_s = 0.009994

    afesim_external = AFESimExternal.from_specification(
        spike_gen_mode="divisive_norm",
        fixed_threshold_vec=None,
        dt=dt_s,
    )
    
    out_external,_,_ = afesim_external((precise_interval_audio, sr))
    
    spikes = TSEvent.from_raster(out_external, dt=dt_s)
    
    # pdb.set_trace()

    xylo.datastructure.append_sample(
        dataset,
        call_dur,
        np.nan,
        precise_start / sr,
        precise_stop / sr,
        "None",
        curr_file,
        CONFIDENCE_THRESHOLD,
        F_LOW,
        F_HIGH,
        BANDPASS_ORDER,
        THRESHOLD_K,
        TIME_PRE,
        TIME_POST,
        sr,
        precise_interval_audio,
        spikes.times,
        spikes.channels,)

    
        
    

