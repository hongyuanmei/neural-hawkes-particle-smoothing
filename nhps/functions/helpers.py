import random
import numpy as np
import torch
import pickle
import os

from nhps.models import nhp
from nhps.io import processors
from nhps.miss import factorized


def load_dataset(**args):
    data_path = os.path.join(args['PathData'], args['Split'] + '.pkl')
    with open(data_path, 'rb') as fp:
        pkl_dev = pickle.load(fp)
    data_dev, data_dev_gold = pkl_dev['seqs_obs'], pkl_dev['seqs']

    if args['Small'] > -1:
        data_dev = data_dev[:args['Small']]
        data_dev_gold = data_dev_gold[:args['Small']]

    #obs_num, unobs_num = pkl_dev['obs_num'], pkl_dev['unobs_num']
    #total_event_num = obs_num + unobs_num
    total_event_num = pkl_dev['total_num']

    return [data_dev, data_dev_gold], [total_event_num]
    #return [data_dev, data_dev_gold], [obs_num, unobs_num, total_event_num]


def propose_particles(**args):

    handler = args['handler']
    handler.print("Propose particles for model {}".format(args['Model']))

    random.seed(args['Seed'])
    np.random.seed(args['Seed'])
    torch.manual_seed(args['Seed'])

    [data_dev, data_dev_gold], [total_event_num] = load_dataset(**args)
    # seq_bases are the x's.
    # We may use miss_mec to generate seq_bases in the future.
    seq_bases = data_dev

    if not args['UseGPU']:
        device = 'cpu'
    else:
        device = 'cuda'

    sampling = 1

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

    if args['Model'] == 'nhps':
        agent = nhp.NeuralHawkes(
            total_num=total_event_num, hidden_dim=args['DimLSTM'],
            device=device, miss_mec=miss_mec
        )
        agent.initBackwardMachine(hidden_dim_back=args['BackDimLSTM'], type_back=args['BackType'],
                                  back_beta=args['BackBeta'])
    elif args['Model'] == 'nhpf':
        agent = nhp.NeuralHawkes(
            total_num=total_event_num, hidden_dim=args['DimLSTM'],
            device=device, miss_mec=miss_mec
        )
    else:
        raise NotImplementedError

    agent.load_state_dict(
        torch.load(args['PathModel'], map_location='cpu') )

    # NOTE: Here I just simply replace neglect_mask with r vector
    # In a more general case mentioned in the paper,
    # r factor should take the whole history into account.
    # But in our experiment, events are censored independently,
    # so the r vector is constant for all the events that're going to be proposed.
    # I set history as None to get the r vector.
    # **Reverse Changes**
    agent.setMaskIntensity(miss_mec.neglect_mask())

    if args['UseGPU']:
        agent.cuda(device)
    agent.eval()

    input_dev = []
    input_dev_with_grountruth = []
    all_weights = list()
    all_particles = list()
    all_log_proposals = list()
    all_num_unobs = list()

    for i_dev, (one_seq_dev, one_seq_dev_gold) in enumerate(zip(data_dev, data_dev_gold)):

        one_seq_dev_augmented = proc.orgSeq(
            one_seq_dev, one_seq_dev_gold[-1]['time_since_start']
        )
        input_dev.append(
            proc.augmentLogProbMissing(
                agent.sample_particles(
                    args['NumParticle'], one_seq_dev_augmented, args['NumUnobserved'],
                    args['Multiplier'],
                    resampling=args['Resampling'],
                    need_eliminate_log_base=args['EliminateBase'] )))

        input_dev_with_grountruth.append(
            proc.processSeq(
                one_seq_dev_gold, n=1, seq_obs=one_seq_dev))

        if (i_dev+1) % args['SizeBatch'] == 0 or \
                (i_dev == len(data_dev)-1 and (len(input_dev)%args['SizeBatch']) > 0):

            r"""
            this part is computing log q(z | x)
            where z is ground truth (similar to train nhps)
            """
            batch_seqs_with_groundtruth = proc.processBatchSeqsWithParticles(
                input_dev_with_grountruth)
            log_proposals, num_unobs = agent(batch_seqs_with_groundtruth, mode=9)
            all_log_proposals.append(log_proposals.detach().cpu().numpy())
            all_num_unobs.append(num_unobs.detach().cpu().numpy())
            input_dev_with_grountruth = list()

            r"""
            this part is computing weights for decoding
            """

            # If resampling is on, we could directly get the weights of particles.
            if args['Resampling']:
                log_weights = np.array([input_[2].cpu().detach().numpy() for input_ in input_dev])
                unnormalized_weights = np.exp(log_weights)
                weights = (unnormalized_weights.T / unnormalized_weights.sum(axis=1)).T
            else:
                # If not, we could use our old method.
                batch_seqs_dev = proc.processBatchSeqsWithParticles(input_dev)
                weights, _ = agent(batch_seqs_dev, mode=4)
                weights = weights.detach().cpu().numpy()

            weights_orders = np.argsort(-weights, axis=1)

            for i_batch, [event_, dtime_, _, _, len_seq_, _, _, _, _, _, _, _] in enumerate(input_dev):
                particles = list()
                all_particles.append(particles)
                for particle_idx in weights_orders[i_batch]:
                    event, dtime, len_seq = event_[particle_idx], \
                                            dtime_[particle_idx], len_seq_[particle_idx]
                    particles.append(proc.getSeq(event.cpu(), dtime.cpu(), len_seq.cpu()))
                all_weights.append(weights[i_batch][weights_orders[i_batch]])

            input_dev = []
            handler.print("Proposing {}-th {} seq".format(i_dev+1, args['Split']))

    all_weights = np.array(all_weights)
    all_log_proposals = np.concatenate(all_log_proposals)
    all_num_unobs = np.concatenate(all_num_unobs)

    # Note that particles in each batch are ordered by the their weights inversely.
    return all_weights, all_particles, seq_bases, all_log_proposals, all_num_unobs
