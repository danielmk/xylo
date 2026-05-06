# -*- coding: utf-8 -*-
"""
nn.modules.LIFTorch -> has w_rec
"""

from rockpool.nn.networks import SynNet
import matplotlib.pyplot as plt

net = SynNet(
    n_channels =16,
    n_classes =2,
    size_hidden_layers = [140, 40, 40, 40, 40, 40],  # Best Performance from the paper
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output='vmem'  # Optional; for training
    )


plt.rcParams["figure.figsize"] = [12, 4]
plt.rcParams["figure.dpi"] = 300

# - Plot the synaptic time constants for each layer
for lyr in net.lif_names[:-1]:
    plt.plot(net.seq[lyr].tau_syn / 1e-3, label = f"Layer {lyr}")
    
plt.xlabel('Neuron ID')
plt.ylabel('Synaptic time constant (ms)')
plt.legend()