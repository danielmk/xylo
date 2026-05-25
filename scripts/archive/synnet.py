#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:06:24 2026

@author: daniel-muller
"""

from rockpool.nn.networks import SynNet

net = SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [140, 40, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    )

print(net)
