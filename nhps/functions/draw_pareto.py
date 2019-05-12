# -*- coding: utf-8 -*-
# !/usr/bin/python
"""

Draw pareto curves to compare NHPF and NHPS

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

#
import argparse
__author__ = 'Hongyuan Mei'


def draw_pareto(Path, Split, x_min=None, x_max=None, y_min=None, y_max=None):
    Path = os.path.abspath(Path)

    r"""
    check if PF and/or PS exist
    only read the log and model that exists
    """

    path_results = os.path.join(
        Path, '{}.results.pkl'.format(Split))

    with open(path_results, 'rb') as f:
        results = pickle.load(f)

    assert results['MBR'] is not None, 'MBR missing'
    assert results['MAP'] is not None, 'MAP missing'

    if True:
        def clear_sub_points(method):
            if isinstance(method, dict) and 'nhps' in method:
                max_epoch = max(method['nhps'].keys())
                method['nhps'] = {
                    max_epoch: method['nhps'][max_epoch]
                }
        for method_name in ['MBR', 'MAP']:
            clear_sub_points(results[method_name])

    # Engineering hack: Only draw pareto for the first max_lim models!
    max_lim = 10
    nhps_ids = list(results['MBR']['nhps'].keys())
    nhps_ids.sort()
    new_dict = dict()
    for key in nhps_ids[:max_lim]:
        new_dict[key] = results['MBR']['nhps'][key]
    results['MBR']['nhps'] = new_dict

    print("Results of both methods collected, draw figures")

    drawer = drawers.Drawer()

    # find the dataset name
    folder_names = Path.split(os.sep) + ['dataset']
    if 'logs' in folder_names:
        tag_data = folder_names[folder_names.index('logs')+1]
        print('dataset {} found.'.format(tag_data))
    else:
        tag_data = 'dataset'

    drawer.setResults(Path, results)
    drawer.draw("{}_{}".format(Split, tag_data), x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def main():

    parser = argparse.ArgumentParser(description='Testing model ...')
    parser.add_argument(
        '-p', '--Path', required=True,
        help='Path to dir of the saved logs and models'
    )
    parser.add_argument(
        '-s', '--Split', default='dev', choices=['dev', 'test'],
        help='Evaluate on dev or test data?'
    )
    args = parser.parse_args()
    dict_args = vars(args)
    draw_pareto(dict_args['Path'], dict_args['Split'])


if __name__ == "__main__":
    main()
