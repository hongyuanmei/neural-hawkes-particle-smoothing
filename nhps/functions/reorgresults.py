# -*- coding: utf-8 -*-
# !/usr/bin/python
"""

reorg results pikle
switch the level of MAP/MBR and nhpf/nhps

@author: hongyuan
"""

import pickle
import time
import numpy
import os
import sys
import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.autograd import Variable

path_results = './taxiresults'

with open(
    os.path.join(path_results, 'dev.results.beforereorg.pkl'), 'rb') as f:
    results = pickle.load(f)

new_results = {
    'MAP': {
        'nhpf': {}, 'nhps': {}
    },
    'MBR': {
        'nhpf': {}, 'nhps': {}
    }
}

new_results['MAP']['nhpf'] = results['nhpf']['MAP']
new_results['MBR']['nhpf'] = results['nhpf']['MBR']

list_ids = sorted(list(results['nhps'].keys()))

for id in list_ids:

    new_results['MAP']['nhps'][id] = results['nhps'][id]['MAP']
    new_results['MBR']['nhps'][id] = results['nhps'][id]['MBR']

with open(os.path.join(path_results, 'dev.results.pkl'), 'wb') as f:
    pickle.dump(new_results, f)
