# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:05:40 2026

@author: Daniel
"""
import numpy as np
import torch
from rockpool.timeseries import TSEvent

rng = np.random.default_rng()



def build_all_rasters(train, t_stop, dt, size_in):
    n = train.spike_times.nrows
    n_steps = int(t_stop / dt)

    rasters_np = np.zeros((n, n_steps, size_in), dtype=np.float32)

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

def build_all_labels(train, species, t_stop, dt, size_out, label_amplitude=1.0):
    n = train.samples.nrows
    n_steps = int(t_stop / dt)

    labels = torch.zeros((n, n_steps, size_out), dtype=torch.float32)

    for i, sample in enumerate(train.samples):
        if species[i] == "None":
            continue

        start = int(1 / dt)
        stop = start + int(sample["call_duration"] / dt)
        labels[i, start:stop, 0] = label_amplitude

    return labels
