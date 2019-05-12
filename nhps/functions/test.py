# -*- coding: utf-8 -*-
# !/usr/bin/python
"""

Test models

@author: hongyuan
"""

import pickle
import os
import argparse
import json

from nhps.io import processors
from nhps.functions.test_helper import run_complete
from mp_manager import Manager

__author__ = 'Hongyuan Mei'


def main():
    parser = argparse.ArgumentParser(description='Testing model ...')
    parser.add_argument(
        '-p', '--Path', required=True,
        help='Path to dir of nhps models'
    )
    parser.add_argument(
        '-nu', '--NumUnobserved', required=True, type=int,
        help='How many unobserved events in each seq at MOST when PROPOSING? e.g. 1, 2, 3, ...'
    )
    parser.add_argument(
        '-mu', '--Multiplier', default=1, type=int,
        help='N = O(I) and what is the constant?'
    )
    parser.add_argument(
        '-np', '--NumParticle', default=1, type=int,
        help='Num of particles to use?'
    )
    parser.add_argument(
        '-c', '--Cost', default=1.0, type=float,
        help='What is (per operation) cost for deletion/insertion?'
    )
    parser.add_argument(
        '-mc', '--MultiplierCost', default=2.0, type=float,
        help='What is the multiplier used to increase cost?'
    )
    parser.add_argument(
        '-nc', '--NumCost', default=1, type=int,
        help='How many cost values to test? Seq of costs will start from Cost and keep incresing.'
    )
    parser.add_argument(
        '-s', '--Split', default='dev', choices=['dev', 'test'],
        help='Evaluate on dev or test data?'
    )
    parser.add_argument(
        '-sb', '--SizeBatch', default=1, type=int,
        help='Size of mini-batch'
    )
    parser.add_argument(
        '-gpu', '--UseGPU', action='store_true',
        help='Use GPU?'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int,
        help='Random seed. e.g. 12345'
    )
    parser.add_argument(
        '-sm', '--Small', default=-1, type=int,
        help='Use a small dataset for debug. Set as -1 to turn off.'
    )
    parser.add_argument(
        '-mi', '--MaxIter', default=5, type=int,
        help='Max iterations (necessary) for consensus decoding.'
    )
    parser.add_argument(
        '-ppd', '--ProcessPerDevice', default=1, type=int,
        help='Maximum number of GPU tasks per device at the same time.'
             + ' Limited by GPU memory and cuda capability. The default is 1.'
    )
    parser.add_argument(
        '-vd', '--VisibleDevice', nargs='*', type=int,
        help='Set the visible devices. All the devices are visible by default.'
    )
    parser.add_argument(
        '-mp', '--MultiProcess', default=None, type=int,
        help='# of processes. Default: automatically adjust to the # of CPU cores.'
    )
    parser.add_argument(
        '-rs', '--Resampling', action='store_true',
        help='Apply resampling.')
    parser.add_argument(
        '-it', '--Intermediate', action='store_true',
        help='Include all the intermediate models.'
    )
    parser.add_argument(
        '-eb', '--EliminateBase', action='store_true',
        help='Adjust magnitude of log weight during SMC to avoid underflow'
    )

    args = parser.parse_args()
    dict_args = vars(args)

    dict_args['Path'] = os.path.abspath(dict_args['Path'])
    path_log = os.path.join(dict_args['Path'], 'log.txt')
    # load saved args in log.txt
    saved_args = processors.LogReader(path_log).getArgs()
    for argname in saved_args:
        if argname not in dict_args:
            dict_args[argname] = saved_args[argname]
    if '' in dict_args:
        del dict_args['']
    dict_args['PathData'] = os.path.abspath(os.path.join(dict_args['Path'], '..', '..', '..',
                                                         '..', 'data', dict_args['Dataset']))

    # multiprocess manager
    manager = Manager(max_process_per_device=dict_args['ProcessPerDevice'],
                      max_processes=dict_args['MultiProcess'],
                      visible_devices=dict_args['VisibleDevice'])

    # following codes for nhpf model

    arg_nhpf = dict_args.copy()
    arg_nhpf['Model'] = 'nhpf'
    arg_nhpf['PathModel'] = os.path.abspath(os.path.join(dict_args['Path'], '..', 'saved_model'))
    identifier_nhpf = ['nhpf']
    # the identifier in arg_nhpf is used to distinguish print info
    # the identifier in manager is used to distinguish returned results
    arg_nhpf['Identifier'] = 'nhpf'
    manager.add_task(identifier_nhpf, run_complete, arg_nhpf)

    # following codes for nhps models

    path_intermediate = os.path.join(dict_args['Path'], 'inter')
    inter_models = list()
    for file_name in os.listdir(path_intermediate):
        if 'model_nhps_' in file_name:
            model_id = int(file_name.split('_')[-1])
            inter_models.append([model_id, file_name])
    inter_models.sort(key=lambda model_tuple: model_tuple[0])

    # By default, only the last model will be taken into consideration.
    if not dict_args['Intermediate']:
        inter_models = inter_models[-1:]

    # for test split, there is no need to evaluate the intermediate models
    if dict_args['Split'] == 'test':
        inter_models = [inter_models[-1]]

    for model_id, file_name in inter_models:
        identifier = ['nhps', model_id]
        path_model = os.path.join(path_intermediate, file_name)
        arg_nhps = dict_args.copy()
        arg_nhps['Identifier'] = '_'.join([str(item) for item in identifier])
        arg_nhps['PathModel'] = path_model
        manager.add_task(identifier, run_complete, arg_nhps)

    msg, pool_rst = manager.run()

    # Record all results
    results = dict()
    items = ['MBR', 'LogProposal', 'MAP']
    for item in items:
        results[item] = {
            'nhpf': None,
            'nhps': dict()
        }

    for identifier, log_point in pool_rst:
        for item in items:
            curr_level = results[item]
            for level_name in identifier[:-1]:
                curr_level = curr_level[level_name]
            curr_level[identifier[-1]] = log_point[item]

    path_to_results = os.path.join(
        dict_args['Path'], '{}.results.pkl'.format(dict_args['Split']))
    with open(path_to_results, 'wb') as fp:
        pickle.dump(results, fp)
    path_to_msg = os.path.join(dict_args['Path'], '{}.msg.json'.format(dict_args['Split']))
    with open(path_to_msg, 'w') as fp:
        json.dump(msg, fp)
    message = "Results saved to {}".format(path_to_results)
    print(message)


if __name__ == "__main__":
    main()
