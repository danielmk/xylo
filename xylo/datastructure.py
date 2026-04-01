# -*- coding: utf-8 -*-
"""
Utilities for creating an empty PyTables HDF5 dataset
for machine learning training data.

Schema:
    /dataset
        ├── metadata        (Table)
        ├── audio           (EArray)
        ├── spike_times     (VLArray)
        └── spike_channels  (VLArray)
"""

import tables
import numpy as np


class SampleMeta(tables.IsDescription):
    """
    Fixed-size per-sample metadata.
    """
    on_time    = tables.Float64Col()
    off_time   = tables.Float64Col()
    confidence = tables.Float32Col()

    t_start = tables.Float64Col()
    t_stop  = tables.Float64Col()

    quality = tables.BoolCol()

    species  = tables.StringCol(32)
    filename = tables.StringCol(128)
    
    sr = tables.Float64Col()


def create_empty_dataset(
    h5_path: str,
    audio_length: int,
    group_name: str = "train",
    audio_dtype=tables.Float32Atom(),
    compression_level: int = 5,
    compression_lib: str = "blosc",
):
    """
    Create an empty HDF5 file with the standard dataset layout.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file to create.
        Existing files will be overwritten.

    audio_length : int
        Length (number of samples) of each raw audio array.

    group_name : str, optional
        Name of the top-level group (default: "dataset").

    audio_dtype : tables.Atom, optional
        Atom type used for storing audio (default: Float32).

    compression_level : int, optional
        Compression level, 0–9 (default: 5).

    compression_lib : str, optional
        Compression library (default: "blosc").
    """

    filters = tables.Filters(
        complevel=compression_level,
        complib=compression_lib
    )

    with tables.open_file(h5_path, mode="w") as h5:
        # Create top-level group
        group = h5.create_group("/", group_name)

        # Metadata table
        h5.create_table(
            group,
            "metadata",
            description=SampleMeta,
            filters=filters
        )

        # Raw audio: growable array (one row per sample)
        h5.create_earray(
            group,
            "audio",
            atom=audio_dtype,
            shape=(0, audio_length),
            filters=filters
        )

        # Variable-length spike information
        h5.create_vlarray(
            group,
            "spike_times",
            atom=tables.Float64Atom(),
            filters=filters
        )

        h5.create_vlarray(
            group,
            "spike_channels",
            atom=tables.Int16Atom(),
            filters=filters
        )


def append_sample(
    h5file: tables.File,
    *,
    on_time: float,
    off_time: float,
    confidence: float,
    species: str,
    filename: str,
    t_start: float,
    t_stop: float,
    quality: float,
    audio: np.ndarray,
    spike_times: np.ndarray,
    spike_channels: np.ndarray,
    group_name: str = "dataset",
):
    """
    Append a single sample to an existing dataset.

    Parameters
    ----------
    h5file : tables.File
        An open PyTables file (mode='a').

    All other parameters correspond to one sample.
    """

    group = h5file.get_node(f"/{group_name}")

    meta = group.metadata
    audio_arr = group.audio
    spike_times_arr = group.spike_times
    spike_channels_arr = group.spike_channels

    # --- Validation ---------------------------------------------------------

    if audio.ndim != 1:
        raise ValueError("audio must be a 1D array")

    if audio.shape[0] != audio_arr.shape[1]:
        raise ValueError(
            f"audio length {audio.shape[0]} does not match "
            f"dataset audio length {audio_arr.shape[1]}"
        )

    if spike_times.shape != spike_channels.shape:
        raise ValueError("spike_times and spike_channels must have the same shape")

    spike_times = np.asarray(spike_times, dtype=np.float64)
    spike_channels = np.asarray(spike_channels, dtype=np.int16)
    audio = np.asarray(audio, dtype=audio_arr.atom.dtype)

    # --- Append metadata ----------------------------------------------------

    row = meta.row
    row["on_time"] = on_time
    row["off_time"] = off_time
    row["confidence"] = confidence
    row["species"] = species
    row["filename"] = filename
    row["t_start"] = t_start
    row["t_stop"] = t_stop
    row["quality"] = quality
    row.append()
    meta.flush()

    # --- Append arrays ------------------------------------------------------

    audio_arr.append(audio[np.newaxis, :])
    spike_times_arr.append(spike_times)
    spike_channels_arr.append(spike_channels)
