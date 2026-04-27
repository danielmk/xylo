# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:34:58 2026

@author: Daniel
"""

import tables

dataset_path = r'Y:\danielmk\okeon\dataset_split.h5'

dataset = tables.open_file(dataset_path, mode="r")