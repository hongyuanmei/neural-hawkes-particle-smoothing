# -*- coding: utf-8 -*-
# !/usr/bin/python
"""

Train neural Hawkes particle smoothing

@author: hongyuan
"""

import pickle
import time
import numpy
import random
import os
import datetime
from itertools import chain
import argparse
import torch
import torch.optim as optim

from nhps.models import nhp
from nhps.io import processors
from nhps.miss import factorized
from nhps.io.processors import LogReader

#
__author__ = 'Hongyuan Mei'

def run_complete(args):

    r"""
    the training of PS should start from a trained p(x,z)
    i.e. PS should read PF's trained model
    PS should save the model into the SAME folder with PF
    so when evaluating, we just need to pass one folder into the code
    and it will get results of both PF and PS for the same seq of penalties
    and draw figures
    """

    assert args['NumUnobserved'] > 0, "hidden events needed"
    assert args['NumPartition'] > 0, "partitions needed"
    assert args['NumParticle'] > 0, "particles needed"

    random.seed(args['Seed'])
    numpy.random.seed(args['Seed'])
    torch.manual_seed(args['Seed'])

    with open(os.path.join(args['PathData'], 'train.pkl'), 'rb') as f:
        pkl_train = pickle.load(f)
    with open(os.path.join(args['PathData'], 'dev.pkl'), 'rb') as f:
        pkl_dev = pickle.load(f)

    learning_rate = args['LearnRate']

    data = pkl_train['seqs']
    data_dev = pkl_dev['seqs']

    r"""
    this data format may need be changed --- there is only one total num now!!!
    """
    #obs_num, unobs_num = pkl_train['obs_num'], pkl_train['unobs_num']
    #total_event_num = obs_num + unobs_num

    total_event_num = pkl_train['total_num']

    r"""
    For ease, when args['Gamma'] >= 0.99,
    we only use inclusive KL div
    so we set args['NumUnobserved'] to be 0
    to save time from proposing
    """
    if args['Gamma'] >= 0.99:
        args['NumUnobserved'] = 0

    agent = nhp.NeuralHawkes(
        total_num=total_event_num, hidden_dim=args['DimLSTM'],
        device='cuda' if args['UseGPU'] else 'cpu',
    )

    agent.load_state_dict(torch.load(args['PathOldModel'], map_location='cpu'))

    if args['UseGPU']:
        agent.cuda()

    agent.initBackwardMachine(
        type_back=args['BackType'],
        hidden_dim_back=args['BackDimLSTM'],
        back_beta=args['BackBeta']
    )

    path_intermediate = args['PathInter']

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
    we only update parameters that are only related to right2left machine
    """
    optimizer = optim.Adam(
        agent.right2left_machine.parameters(), lr=learning_rate
    )

    print("Start training ... ")
    total_logP_best = -1e6
    total_KL_best = 1e6
    avg_dis_best = 1e6
    time0 = time.time()

    episodes = []
    total_rewards = []

    max_episode = args['MaxEpoch'] * len(data)
    report_gap = args['TrackPeriod']

    time_sample = 0.0
    time_train_only = 0.0
    time_dev_only = 0.0
    input = []
    input_particles = []

    for episode in range(max_episode):

        idx_seq = episode % len(data)
        idx_epoch = episode // len(data)
        one_seq = data[ idx_seq ]

        r"""
        throw the complete seq through MissMec
        to get partitions
        block of code that is similar to proc.processSeq( one_seq )
        """
        input.append( proc.processSeq( one_seq, n=args['NumPartition'] ) )

        r"""
        throw a partitioned observed seq x through sampler
        to get particles
        block of code that is similar to line-78 to line-83 in helpers
        """
        partition = proc.processSeq( one_seq, n=1 )
        one_seq_obs = proc.orgSeq(
            proc.getSeq(
                partition[5][0, :], partition[6][0, :], partition[5].size(1) - 2 ),
            one_seq[-1]['time_since_start'])

        time_sample_0 = time.time()
        input_particles.append(
            proc.augmentLogProbMissing(
                agent.sample_particles(args['NumParticle'], one_seq_obs, args['NumUnobserved'])))
        time_sample += (time.time() - time_sample_0)

        if len(input) >= args['SizeBatch']:

            r"""
            two batches of seqs:
            1. batch of partitions --- for inclusive KL
            2. batch of particles --- for exclusive KL
            """

            batch_partitions = proc.processBatchSeqsWithParticles( input )
            batch_particles = proc.processBatchSeqsWithParticles( input_particles )

            agent.train()
            time_train_only_0 = time.time()

            r"""
            KL divergence that we ought to minimize !!
            refer to paper for details
            """

            inc_KL, _ = agent( batch_partitions, mode=7 )
            (inc_KL * args['Gamma']).backward()

            exc_KL, _ = agent( batch_particles, mode=8 )
            (exc_KL * (1.0 - args['Gamma']) ).backward()

            optimizer.step()
            optimizer.zero_grad()

            time_train_only += (time.time() - time_train_only_0)

            input = []
            input_particles = []

            if episode % report_gap == report_gap - 1:

                r"""
                for eval, we early stop on LOSS = KL divergence
                """

                time1 = time.time()
                time_train = time1 - time0
                time0 = time1

                print("Validating at episode {} ({}-th seq of {}-th epoch)".format(
                    episode, idx_seq, idx_epoch))
                total_logP = 0.0
                total_KL_inc = 0.0
                total_KL_exc = 0.0
                total_KL = 0.0
                total_num_inc = 0.0
                total_num_exc = 0.0
                #total_num = 0.0

                input_dev = []
                input_particles_dev = []
                agent.eval()

                for i_dev, one_seq_dev in enumerate(data_dev):

                    input_dev.append(
                        proc.processSeq( one_seq_dev, n=args['NumPartition'] ) )

                    partition_dev = proc.processSeq( one_seq_dev, n=1 )
                    one_seq_obs_dev = proc.orgSeq(
                        proc.getSeq(
                            partition_dev[5][0, :], partition_dev[6][0, :],
                            partition_dev[5].size(1) - 2 ),
                        one_seq_dev[-1]['time_since_start'] )

                    time_sample_0 = time.time()
                    input_particles_dev.append(
                        proc.augmentLogProbMissing(
                            agent.sample_particles(
                                args['NumParticle'], one_seq_obs_dev, args['NumUnobserved']) ) )
                    time_sample += (time.time() - time_sample_0)

                    if (i_dev+1) % args['SizeBatch'] == 0 or \
                            (i_dev == len(data_dev)-1 and (len(input_dev)%args['SizeBatch']) > 0):

                        batch_partitions_dev = proc.processBatchSeqsWithParticles(
                            input_dev )
                        batch_particles_dev = proc.processBatchSeqsWithParticles(
                            input_particles_dev )

                        time_dev_only_0 = time.time()

                        inc_KL, num_events_inc = agent(
                            batch_partitions_dev, mode=7 )
                        exc_KL, num_events_exc = agent(
                            batch_particles_dev, mode=8 )

                        time_dev_only = time.time() - time_dev_only_0

                        total_KL_inc += float(inc_KL.data.sum() )
                        total_KL_exc += float(exc_KL.data.sum() )

                        total_num_inc += float(
                            num_events_inc.data.sum() / (
                                args['NumPartition'] * args['SizeBatch'] * 1.0 ) )
                        total_num_exc += float(
                            num_events_exc.data.sum() / (
                                args['NumParticle'] * args['SizeBatch'] * 1.0 ) )

                        input_dev = []
                        input_particles_dev = []

                total_KL_inc /= max(total_num_inc, 1.0)
                total_KL_exc /= max(total_num_exc, 1.0)
                total_KL = args['Gamma'] * total_KL_inc + (1.0 - args['Gamma']) * total_KL_exc

                message = "Episode {} ({}-th seq of {}-th epoch), KL divergence is {:.4f} with inclusive KL {:.4f} and exclusive KL {:.4f} and gamma {:.2f}".format(
                    episode, idx_seq, idx_epoch, total_KL,
                    total_KL_inc, total_KL_exc, args['Gamma'] )
                logger.checkpoint(message)
                print(message)

                updated = None
                if total_KL < total_KL_best:
                    total_KL_best = total_KL
                    updated = True
                    episode_best = episode
                else:
                    updated = False
                message = "Current best KL is {:.4f} (updated at episode {})".format(
                    total_KL_best, episode_best )

                if updated:
                    message += ", best updated at this episode"
                    torch.save(agent.state_dict(), args['PathSave'])
                logger.checkpoint(message)

                #no matter useful or not, we save this intermediate model
                #for possible future use
                torch.save(
                    agent.state_dict(),
                    os.path.join(path_intermediate, 'model_nhps_{}'.format(episode+1))
                )

                print(message)
                episodes.append(episode)

                time1 = time.time()
                time_dev = time1 - time0
                time0 = time1
                message = "time to train {} episodes is {:.2f} and time for dev is {:.2f}".format(
                    report_gap, time_train, time_dev )
                message += "\ntime to sample (train + dev) for {} episodes is {:.2f}".format(
                    report_gap, time_sample)

                time_sample, time_train_only = 0.0, 0.0
                time_dev_only = 0.0
                #
                logger.checkpoint(message)
                print(message)
    message = "training finished"
    logger.checkpoint(message)
    print(message)

    #delete all the intermediate models that saved after episode_best
    print("cleaning intermediate models : deleting those after {}-episode ...".format(episode_best))
    files_to_delete = []
    for subdir, dirs, files in os.walk(path_intermediate):
        for file in files:
            if 'model_nhps_' in file and int(file.split('_')[-1]) > episode_best:
                files_to_delete.append( os.path.join(subdir, file) )
    print("there are {} such files".format(len(files_to_delete)))
    for file_to_delete in files_to_delete:
        os.remove(file_to_delete)
    print("done")

def main():

    parser = argparse.ArgumentParser(description='Trainning model ...')
    parser.add_argument(
        '-pn', '--PathNHPF', required=True,
        help='Path of saved neural Hawkes?'
    )
    parser.add_argument(
        '-bt', '--BackType', default='add',
        choices=['sep', 'add', 'mul'],
        help='Type of right-to-left parametrization'
    )
    parser.add_argument(
        '-bd', '--BackDimLSTM', default=16, type=int,
        help='Dimension of LSTM for backward (right2left)?'
    )
    parser.add_argument(
        '-m', '--Multiplier', default=1, type=int,
        help='N = O(I) and what is the constant?'
    )
    parser.add_argument(
        '-be', '--BackBeta', default=1.0, type=float,
        help='Beta for Back LSTM'
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
        '-me', '--MaxEpoch', default=20, type=int,
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
        '-nu', '--NumUnobserved', default=0, type=int,
        help='How many unobserved events in each seq at MOST when PROPOSING?, e.g. 0, 1, 2, ...'
    )
    parser.add_argument(
        '-npn', '--NumPartition', default=1, type=int,
        help='Num of partitions given complete sequence x and z?'
    )
    parser.add_argument(
        '-np', '--NumParticle', default=1, type=int,
        help='Num of particles given observed sequence x?'
    )
    parser.add_argument(
        '-ga', '--Gamma', default=1.0, type=float,
        help='How much weight (0.0 -- 1.0) on inclusive KL div vs. exclusive KL div'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int,
        help='Random seed. e.g. 12345'
    )

    args = parser.parse_args()
    dict_args = vars(args)
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    dict_args['Version'] = torch.__version__
    dict_args['ID'] = id_process
    dict_args['TIME'] = time_current
    dict_args['PathNHPF'] = os.path.abspath(dict_args['PathNHPF'])

    nhpf_path = dict_args['PathNHPF']
    assert os.path.isdir(nhpf_path)

    nhpf_log = os.path.join(nhpf_path, 'log.txt')
    log_reader = LogReader(nhpf_log)
    for key, value in log_reader.getArgs().items():
        if key not in dict_args:
            dict_args[key] = value

    # format: [arg name, name used in path]
    args_used_in_name = [
        ['BackType', 'bt'],
        ['BackDimLSTM', 'dim'],
        ['SizeBatch', 'batch'],
        ['NumPartition', 'npn'],
        ['NumParticle', 'np'],
        ['Gamma', 'ga'],
        ['LearnRate', 'lr'],
        ['Seed', 'seed'],
    ]
    folder_name = list()
    for arg_name, rename in args_used_in_name:
        folder_name.append('{}-{}'.format(rename, dict_args[arg_name]))
    folder_name = '-'.join(folder_name)
    folder_name = '{}_{}'.format(folder_name, id_process)

    path_log = os.path.join(nhpf_path, folder_name)
    os.makedirs(path_log)

    file_log = os.path.join(path_log, 'log.txt')
    file_model = os.path.join(os.path.abspath(path_log), 'saved_model')

    dict_args['PathLog'] = file_log
    dict_args['PathData'] = os.path.abspath(
        os.path.join(path_log, '..', '..', '..', '..', 'data', dict_args['Dataset']))
    dict_args['PathSave'] = file_model
    dict_args['PathOldModel'] = os.path.join(nhpf_path, 'saved_model')
    dict_args['PathInter'] = os.path.join(path_log, 'inter')
    os.makedirs(dict_args['PathInter'])
    dict_args['Model'] = 'nhps'

    if '' in dict_args:
        del dict_args['']

    run_complete(dict_args)


if __name__ == "__main__":
    main()
