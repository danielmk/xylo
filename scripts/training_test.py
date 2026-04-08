#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:49:15 2026

@author: daniel-muller
"""

from rockpool.nn.networks import SynNet
import tables
from torch.optim import Adam, SGD
from torch.nn import MSELoss
import numpy as np
from rockpool.timeseries import TSEvent
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio

# dataset_path = r'/flash/FukaiU/danielmk/dataset.h5'

dataset_path = r'/flash/FukaiU/danielmk/dataset.h5'

dataset = tables.open_file(dataset_path, mode='r+')

net = SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [140, 40, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    )

print(net)

# - Get the optimiser functions
optimizer = Adam(net.parameters().astorch(), lr=1e-4)

# - Loss function
loss_fun = MSELoss()

"""SELECT RANDOM SAMPLES"""
n = dataset.root.train.samples.shape[0]
k = 100           # number to select

indices = np.random.choice(n, size=k, replace=False)

test_index = 7

sr=dataset.root.train.samples[test_index]['sr']
D = librosa.amplitude_to_db(np.abs(librosa.stft(dataset.root.train.audio[indices[test_index]])), ref=np.max)
fig, ax = plt.subplots(1, 1)
img = librosa.display.specshow(D, y_axis='linear', x_axis='s', sr=sr, ax=ax)

event = TSEvent(
    times=dataset.root.train.spike_times[indices[test_index]],
    channels=dataset.root.train.spike_channels[indices[test_index]],
    t_stop=2.504
)

plt.figure()
event.plot(marker='|', alpha=0.8)

output, state, rec = net(event, record=True)