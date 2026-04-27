# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:57:13 2026

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

src_path = r"Y:\danielmk\okeon\dataset.h5"
dst_path = r"Y:\danielmk\okeon\dataset_split.h5"

with tables.open_file(src_path, mode="r") as src:

    q = src.root.train.quality_rating[:]        # (N,)
    species = src.root.train.samples.col("species")  # (N,)

    # Decode bytes → str if needed
    species = np.array([s.decode() if isinstance(s, bytes) else s for s in species])

    # Keep sample if:
    # quality != 0 OR species == "None"


    is_noise = (species == "None")
    is_signal = (~is_noise) & (q > 1)
    
    valid_mask = is_noise | is_signal

    valid_indices = np.where(valid_mask)[0]


rng = np.random.default_rng(seed=42)
rng.shuffle(valid_indices)

n_total = len(valid_indices)
n_train = int(0.9 * n_total)

train_idx = valid_indices[:n_train]
test_idx  = valid_indices[n_train:]

with tables.open_file(src_path, mode="r") as src, \
     tables.open_file(dst_path, mode="w") as dst:

    for split_name, split_idx in [("train", train_idx), ("test", test_idx)]:
        grp = dst.create_group("/", split_name)

        # Audio
        audio_src = src.root.train.audio
        audio_dst = dst.create_earray(
            grp, "audio",
            atom=audio_src.atom,
            shape=(0, audio_src.shape[1]),
            filters=audio_src.filters
        )

        # Quality rating
        q_dst = dst.create_earray(
            grp, "quality_rating",
            atom=src.root.train.quality_rating.atom,
            shape=(0,),
            filters=src.root.train.quality_rating.filters
        )

        # Samples table
        samples_dst = dst.create_table(
            grp, "samples",
            description=src.root.train.samples.description,
            filters=src.root.train.samples.filters
        )

        # VLArrays
        sc_dst = dst.create_vlarray(
            grp, "spike_channels",
            atom=src.root.train.spike_channels.atom,
            filters=src.root.train.spike_channels.filters
        )

        st_dst = dst.create_vlarray(
            grp, "spike_times",
            atom=src.root.train.spike_times.atom,
            filters=src.root.train.spike_times.filters
        )
    
        for i in split_idx:
            audio_dst.append(src.root.train.audio[i:i+1])
            q_dst.append(src.root.train.quality_rating[i:i+1])
    
            samples_dst.append(src.root.train.samples.read(start=i, stop=i+1))
            sc_dst.append(src.root.train.spike_channels[i])
            st_dst.append(src.root.train.spike_times[i])
    
        samples_dst.flush()