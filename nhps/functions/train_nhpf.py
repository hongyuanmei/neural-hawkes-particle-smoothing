# -*- coding: utf-8 -*-
# !/usr/bin/python
"""

Train neural Hawkes process with complete data

@author: hongyuan
"""

import pickle
import time
import numpy
import random
import os
import datetime
from itertools import chain

import torch
import torch.optim as optim

from nhps.models import nhp
from nhps.io import processors
from nhps.miss import miss_mec, factorized

import argparse
__author__ = 'Hongyuan Mei'


def run_complete(args):

    random.seed(args['Seed'])
    numpy.random.seed(args['Seed'])
    torch.manual_seed(args['Seed'])

    with open(os.path.join(args['PathData'], 'train.pkl'), 'rb') as f:
        pkl_train = pickle.load(f)
    with open(os.path.join(args['PathData'], 'dev.pkl'), 'rb') as f:
        pkl_dev = pickle.load(f)

    learning_rate = args['LearnRate']

    data = pkl_train['seqs']
    #data_dev, data_dev_gold = pkl_dev['seqs_obs'], pkl_dev['seqs']
    data_dev = pkl_dev['seqs']

    #obs_num, unobs_num = pkl_train['obs_num'], pkl_train['unobs_num']
    #total_event_num = obs_num + unobs_num

    total_event_num = pkl_train['total_num']

    hidden_dim = args['DimLSTM']

    agent = nhp.NeuralHawkes(
        total_num=total_event_num, hidden_dim=hidden_dim,
        device='cuda' if args['UseGPU'] else 'cpu'
    )

    if args['UseGPU']:
        agent.cuda()

    sampling = args['Multiplier']

    miss_mec = factorized.FactorizedMissMec(
        device = 'cuda' if args['UseGPU'] else 'cpu',
        config_file = os.path.join(args['PathData'], 'censor.conf')
    )

    proc = processors.DataProcessorNeuralHawkes(
        idx_BOS=total_event_num,
        idx_EOS=total_event_num+1,
        idx_PAD=total_event_num+2,
        miss_mec=miss_mec,
        sampling=sampling,
        device = 'cuda' if args['UseGPU'] else 'cpu'
    )
    logger = processors.LogWriter(args['PathLog'], args)

    r"""
    we only update parameters that are only related to left2right machine
    """
    optimizer = optim.Adam(
        agent.parameters(), lr=learning_rate
    )

    print("Start training ... ")
    total_logP_best = -1e6
    avg_dis_best = 1e6
    episode_best = -1
    time0 = time.time()

    episodes = []
    total_rewards = []

    max_episode = args['MaxEpoch'] * len(data)
    report_gap = args['TrackPeriod']

    time_sample = 0.0
    time_train_only = 0.0
    time_dev_only = 0.0
    input = []

    for episode in range(max_episode):

        idx_seq = episode % len(data)
        idx_epoch = episode // len(data)
        one_seq = data[ idx_seq ]

        #time_sample_0 = time.time()
        input.append( proc.processSeq( one_seq, n=1 ) )
        #time_sample += (time.time() - time_sample_0)

        if len(input) >= args['SizeBatch']:

            batchdata_seqs = proc.processBatchSeqsWithParticles( input )

            agent.train()
            time_train_only_0 = time.time()

            objective, _ = agent( batchdata_seqs, mode=1 )
            objective.backward()

            optimizer.step()
            optimizer.zero_grad()
            time_train_only += (time.time() - time_train_only_0)

            input = []

            if episode % report_gap == report_gap - 1:

                time1 = time.time()
                time_train = time1 - time0
                time0 = time1

                print("Validating at episode {} ({}-th seq of {}-th epoch)".format(
                    episode, idx_seq, idx_epoch))
                total_logP = 0.0
                total_num_token = 0.0

                input_dev = []
                agent.eval()

                for i_dev, one_seq_dev in enumerate(data_dev):

                    input_dev.append(
                        proc.processSeq( one_seq_dev, n=1 ) )

                    if (i_dev+1) % args['SizeBatch'] == 0 or \
                            (i_dev == len(data_dev)-1 and (len(input_dev)%args['SizeBatch']) > 0):

                        batchdata_seqs_dev = proc.processBatchSeqsWithParticles(
                            input_dev )

                        time_dev_only_0 = time.time()
                        objective_dev, num_events_dev = agent(
                            batchdata_seqs_dev, mode=1 )
                        time_dev_only = time.time() - time_dev_only_0

                        total_logP -= float( objective_dev.data.sum() )

                        total_num_token += float(
                            num_events_dev.data.sum() / ( 1 * 1.0 ) )

                        input_dev = []

                total_logP /= total_num_token

                message = "Episode {} ({}-th seq of {}-th epoch), loglik is {:.4f}".format(
                    episode, idx_seq, idx_epoch, total_logP )
                logger.checkpoint(message)
                print(message)

                updated = None
                if total_logP > total_logP_best:
                    total_logP_best = total_logP
                    updated = True
                    episode_best = episode
                else:
                    updated = False
                message = "Current best loglik is {:.4f} (updated at episode {})".format(
                    total_logP_best, episode_best )

                if updated:
                    message += ", best updated at this episode"
                    torch.save(
                        agent.state_dict(), args['PathSave'])
                logger.checkpoint(message)

                print(message)
                episodes.append(episode)

                time1 = time.time()
                time_dev = time1 - time0
                time0 = time1
                message = "time to train {} episodes is {:.2f} and time for dev is {:.2f}".format(
                    report_gap, time_train, time_dev )

                time_sample, time_train_only = 0.0, 0.0
                time_dev_only = 0.0
                #
                logger.checkpoint(message)
                print(message)
    message = "training finished"
    logger.checkpoint(message)
    print(message)


def main():

    parser = argparse.ArgumentParser(description='Trainning model ...')

    parser.add_argument(
        '-ds', '--Dataset', type=str, required=True,
        help='e.g. pilothawkes'
    )
    parser.add_argument(
        '-rp', '--RootPath', type=str,
        help='Root path of project',
        default='../..'
    )
    parser.add_argument(
        '-d', '--DimLSTM', default=16, type=int,
        help='Dimension of LSTM?'
    )
    parser.add_argument(
        '-m', '--Multiplier', default=1, type=int,
        help='N = O(I) and what is the constant?'
    )
    parser.add_argument(
        '-sb', '--SizeBatch', default=50, type=int,
        help='Size of mini-batch'
    )
    parser.add_argument(
        '-tp', '--TrackPeriod', default=5000, type=int,
        help='How many sequences before every checkpoint?'
    )
    parser.add_argument(
        '-me', '--MaxEpoch', default=50, type=int,
        help='Max epoch number of training'
    )
    parser.add_argument(
        '-lr', '--LearnRate', default=1e-3, type=float,
        help='What is the (starting) learning rate?'
    )
    parser.add_argument(
        '-gpu', '--UseGPU', action='store_true',
        help='Use GPU?'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int,
        help='Random seed. e.g. 12345'
    )

    args = parser.parse_args()
    dict_args = vars(args)
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    root_path = os.path.abspath(dict_args['RootPath'])
    dict_args['PathData'] = os.path.join(root_path, 'data', dict_args['Dataset'])
    dict_args['Version'] = torch.__version__
    dict_args['ID'] = id_process
    dict_args['TIME'] = time_current

    # format: [arg name, name used in path]
    args_used_in_name = [
        ['DimLSTM', 'dim'],
        ['SizeBatch', 'batch'],
        ['Seed', 'seed'],
        ['LearnRate', 'lr'],
    ]
    folder_name = list()
    for arg_name, rename in args_used_in_name:
        folder_name.append('{}-{}'.format(rename, dict_args[arg_name]))
    folder_name = '_'.join(folder_name)
    folder_name = '{}_{}'.format(folder_name, id_process)
    print(folder_name)

    path_log = os.path.join(root_path, 'logs', dict_args['Dataset'], folder_name)
    os.makedirs(path_log)

    file_log = os.path.join(path_log, 'log.txt')
    file_model = os.path.join(path_log, 'saved_model')

    dict_args['PathLog'] = file_log
    dict_args['PathSave'] = file_model
    dict_args['Model'] = 'nhpf'

    if '' in dict_args:
        del dict_args['']

    run_complete(dict_args)


if __name__ == "__main__": main()
