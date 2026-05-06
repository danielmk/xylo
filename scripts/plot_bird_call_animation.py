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


dataset_path = r'Y:\danielmk\okeon\dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

high_quality = np.argwhere(dst.root.train.quality_rating.read() == 3)[:, 0]

example_idx = high_quality[2]

sr=44100

audio = dst.root.train.audio[example_idx]

t = np.arange(0, 2.5, 1/sr)

spike_times = dst.root.train.spike_times[example_idx]
spike_channels = dst.root.train.spike_channels[example_idx]

plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(
    3, 1,
    figsize=(16, 9),   # 16:9, safe
    dpi=100            # safe DPI
)


ax[0].plot(t, audio, color='k', linewidth=0.5)
ax[0].set_ylabel("Raw Audio")
ax[0].set_xticklabels([])

D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='s', sr=sr, ax=ax[1])
ax[1].set_xlabel("")
ax[1].set_xticklabels([])

ax[2].scatter(spike_times, spike_channels, marker="|", color='k', alpha=0.8, linewidth=0.5)

ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("# Neuron")
# Exclude long calls
# precise_intervals = precise_intervals[(precise_intervals[:, 1] - precise_intervals[:, 0]) <= 2, :]

for a in ax:
    a.set_xlim((0, 2.5))
    
cursor_lines = [
    a.axvline(0, color="r", lw=2)
    for a in ax
]

plt.tight_layout()


duration = 2.5        # seconds
fps = 30
n_frames = int(duration * fps)

dt = 1.0 / fps


def update(frame):
    t_now = frame * dt
    for line in cursor_lines:
        line.set_xdata([t_now, t_now])
    return cursor_lines



ani = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    blit=True
)

writer = FFMpegWriter(
    fps=30,
    codec="libx264",
    extra_args=[
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    ],
)


ani.save(
    r"C:\Users\Daniel\repos\xylo\results\bird_animation_video.mp4",
    writer=writer,
)


sf.write(
    r"C:\Users\Daniel\repos\xylo\results\bird_call.wav",
    audio,   # NumPy array (float32 or float64 is fine)
    sr,      # sample rate, e.g. 44100
    subtype="FLOAT"
)


