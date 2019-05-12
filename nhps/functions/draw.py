# -*- coding: utf-8 -*-
# !/usr/bin/python
"""

Draw figures to compare performance of different models

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

from nhps.io import processors, drawers

import argparse
__author__ = 'Hongyuan Mei'

def main():

    parser = argparse.ArgumentParser(description='Drawing results ...')
    parser.add_argument(
        '-ppf', '--PathPF', required=True,
        help='Path to dir of the saved results for Particle Filtering'
    )
    parser.add_argument(
        '-pps', '--PathPS', required=True,
        help='Path to dir of the saved results for Particle Smoothing'
    )
    parser.add_argument(
        '-pf', '--PathFigure', default='./', help='Path to the data'
    )
    parser.add_argument(
        '-s', '--Split', default='dev', choices=['dev', 'test'],
        help='Evaluate on dev or test data?'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int,
        help='Random seed. e.g. 12345'
    )
    args = parser.parse_args()
    dict_args = vars(args)
    dict_args['PathPF'] = os.path.abspath(dict_args['PathPF'])
    dict_args['PathPS'] = os.path.abspath(dict_args['PathPS'])
    dict_args['PathFigure'] = os.path.abspath(dict_args['PathFigure'])

    drawer = drawers.Drawer(
        dict_args['PathPF'], dict_args['PathPS'],
        dict_args['Split'], dict_args['PathFigure'])

    drawer.readResults()
    drawer.draw()


if __name__ == "__main__": main()
