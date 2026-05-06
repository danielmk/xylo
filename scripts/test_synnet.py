# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:34:58 2026

@author: Daniel
"""

import tables
import numpy as np
from rockpool.nn.networks import SynNet
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from rockpool.timeseries import TSEvent
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
import sys
import pdb
from pathlib import Path
from xylo import evaluation


"""HYPERPARAMETERS"""
t_stop=2.504
batch_size=64

"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

# dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device("cpu")

dataset_path = r'Y:\danielmk\okeon\dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

test = dst.root.test

q_test = test.quality_rating[:]
species_test = test.samples.col("species")
species_test = np.array([s.decode() if isinstance(s, bytes) else s for s in species_test])

y_true_test = np.zeros((species_test.shape[0]))
y_true_test[species_test=='Ruddy Kingfisher'] = 1

train = dst.root.train

q_train = train.quality_rating[:]
species_train = train.samples.col("species")
species_train = np.array([s.decode() if isinstance(s, bytes) else s for s in species_train])

y_true_train = np.zeros((species_train.shape[0]))
y_true_train[species_train=='Ruddy Kingfisher'] = 1

rng = np.random.default_rng()

net = SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [140, 40, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output='spikes',
    threshold=0.5,
    threshold_out=1.5,
    train_time_constants=True,
    train_threshold=True,
    )

net = net.to(device)

def build_all_rasters(train, t_stop, dt):
    n = train.spike_times.nrows
    n_steps = int(t_stop / dt)

    rasters_np = np.zeros((n, n_steps, net.size_in), dtype=np.float32)

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

def build_all_labels(train, species, t_stop, dt, label_amplitude=1.0):
    n = train.samples.nrows
    n_steps = int(t_stop / dt)

    labels = torch.zeros((n, n_steps, net.size_out), dtype=torch.float32)

    for i, sample in enumerate(train.samples):
        if species[i] == "None":
            continue

        start = int(1 / dt)
        stop = start + int(sample["call_duration"] / dt)
        labels[i, start:stop, 0] = label_amplitude

    return labels

print("Building rasters...")
all_rasters_test = build_all_rasters(test, t_stop, net.dt)

all_rasters_train = build_all_rasters(train, t_stop, net.dt)

n_train = 500

all_rasters_train = all_rasters_train[:n_train, :, :]

y_true_train = y_true_train[:n_train]

# Move **once**
all_rasters_test = all_rasters_test.to(device)

all_rasters_train = all_rasters_train.to(device)

def to_raster(times : list, channels : list, t_start=0.0, t_stop=2.504, dt=0.001):
    
    if len(times) != len(channels): raise ValueError
    
    rasters = []
    for i in range(len(times)):
        event = TSEvent(
            times=times[i],
            channels=channels[i],
            t_stop=t_stop
            )
        raster = event.raster(dt, t_start=t_start, t_stop=t_stop, add_events=True)
        
        rasters.append(raster)
    return torch.Tensor(rasters)

def balanced_accuracy(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # recall
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # specificity

    return 0.5 * (tpr + tnr)

def predict_events(net, rasters):
    output, _, _ = net(rasters, record=False)
    return torch.any(
        output[:, int(1.0 / net.dt):, 0] == 1,
        axis=1
    ).cpu().numpy()


ckpt_dir = Path(r"C:\Users\Daniel\repos\xylo\scripts\checkpoints")

synnet_ckpts = sorted(
    p for p in ckpt_dir.iterdir()
    if p.is_file() and "synnet-long_" in p.name
)


synnet_ckpts = sorted(
    synnet_ckpts,
    key=lambda p: torch.load(p, map_location="cpu").get("epoch", 0)
)

checkpoints = [
    torch.load(path, map_location="cpu")
    for path in synnet_ckpts
]


threshold_grid = np.arange(1.0, 2.1, 0.1)

CONFUSION_KEYS = [
    "tpr", "fnr",
    "tnr", "fpr",
    "precision", "fdr",
    "accuracy", "balanced_accuracy",
    "TP", "TN", "FP", "FN",
]

training_metrics = []
test_metrics = []

epochs = []

loss= []

for ckpt in checkpoints:

    train_ckpt_metrics = {k: [] for k in CONFUSION_KEYS}
    test_ckpt_metrics  = {k: [] for k in CONFUSION_KEYS}
    
    epochs.append(ckpt['epoch'])
    loss.append(ckpt['loss'])

    for thr in threshold_grid:

        net = SynNet(
            n_channels = 16,
            n_classes = 1,
            size_hidden_layers = [140, 40, 40, 40, 40, 40],
            time_constants_per_layer = [2, 2, 4, 4, 8, 8],
            output='spikes',
            threshold=0.5,
            threshold_out=thr,
            # train_time_constants=True,
            # train_threshold=True,
            ).to(device)
        
        net.load_state_dict(ckpt["model_state"])
        net.eval()                # important!
        
        output_train, _, _ = net(all_rasters_train, record=False)
        output_test, _, _ = net(all_rasters_test, record=False)
        
        y_pred_train = predict_events(net, all_rasters_train)
        y_pred_test  = predict_events(net, all_rasters_test)

        train_rates = evaluation.confusion_rates(y_true_train, y_pred_train)
        test_rates = evaluation.confusion_rates(y_true_test, y_pred_test)
        

        # Store everythig
        for k in CONFUSION_KEYS:
            train_ckpt_metrics[k].append(train_rates[k])
            test_ckpt_metrics[k].append(test_rates[k])

        print(
            f"Epoch {ckpt['epoch']:>3} | "
            f"thr={thr:.2f} | "
            f"BA train={train_rates['balanced_accuracy']:.3f}, "
            f"test={test_rates['balanced_accuracy']:.3f} | "
            f"FPR test={test_rates['fpr']:.3f}"
        )


    training_metrics.append(train_ckpt_metrics)
    test_metrics.append(test_ckpt_metrics)


np.savez(
    r"C:\Users\Daniel\repos\xylo\results\synnet-long_threshold_checkpoint_confusion_metric.npz",
    thresholds=threshold_grid,
    loss=loss,
    epochs=epochs,
    training_metrics=training_metrics,
    test_metrics=test_metrics,
    allow_pickle=True
)
