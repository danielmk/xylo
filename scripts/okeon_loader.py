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

datapath = pathlib.Path("Y:\danielmk\okeon")

detections = pd.read_csv(datapath / "2sp_detect_forDaniel.csv")

filename = "CHATANOP_20200508_063000.flac"

location, date, number = filename.split('_')

filepath = datapath / location / date[:4] / filename

data, sr = librosa.load(filepath)

D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)

img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr)

hop_length = 1024

librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
                         x_axis='time')

