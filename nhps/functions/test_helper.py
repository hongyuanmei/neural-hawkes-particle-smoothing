import numpy as np
import pickle
import os

from nhps.functions import helpers
from nhps.distance import DistanceRecorder, MapDecoder,\
    ConsensusDecoder, remove_bases_for_test, LogProbRecorder


def run_complete(**args):
    handler = args['handler']

    cache_name = '{}_{}.{}.pkl'.format(args['Identifier'], args['NumUnobserved'], args['Split'])
    cache_path = os.path.join(args['Path'], 'particle_cache', cache_name)
    if os.path.exists(cache_path):
        all_weights, all_particles, bases, all_log_proposals, all_num_unobs = pickle.load(open(cache_path, 'rb'))
    else:
        if not os.path.exists(os.path.dirname(cache_path)):
            os.mkdir(os.path.dirname(cache_path))
        if args['UseGPU']:
            all_weights, all_particles, bases, all_log_proposals, all_num_unobs = handler.use_device_to_run(
                helpers.propose_particles, args)
        else:
            all_weights, all_particles, bases, all_log_proposals, all_num_unobs = helpers.propose_particles(**args)
        pickle.dump([all_weights, all_particles, bases, all_log_proposals, all_num_unobs], open(cache_path, 'wb'))

    average_ess = np.average(all_weights.sum(axis=1)**2 / (all_weights ** 2).sum(axis=1))
    handler.print('Effective sampling size: {}'.format(average_ess))

    #[_, data_dev_gold], [obs_num, unobs_num, _] = helpers.load_dataset(**args)
    #n_types = obs_num + unobs_num

    [_, data_dev_gold], [n_types] = helpers.load_dataset(**args)

    # Remove x's from x \sqcup z's
    all_particles, data_dev_gold = remove_bases_for_test(all_particles, data_dev_gold, bases)

    del_cost_values = np.full(shape=[args['NumCost']], fill_value=args['MultiplierCost'], dtype=np.float32)
    del_cost_values = del_cost_values.cumprod() * args['Cost'] / args['MultiplierCost']
    distance_recorder = DistanceRecorder(del_cost_values, n_types)
    logprob_recorder = LogProbRecorder(all_log_proposals, all_num_unobs)

    results = dict()

    decoders = {
        'MAP': MapDecoder(del_cost_values, n_types),
        'MBR': ConsensusDecoder(del_cost_values, n_types, args['MaxIter']),
    }

    for decoder_name, decoder in decoders.items():
        print_period = 100
        distance_recorder.reset()
        for seq_idx, [particles, gold_answer, weights] in\
                enumerate(zip(all_particles, data_dev_gold, all_weights)):
            decoded = decoder.decode(particles, weights)
            distance_recorder.record(gold_answer, decoded)
            if (seq_idx - 1) % print_period == 0:
                handler.print('{} decoding: {}'.format(args['Identifier'], seq_idx-1))
        results[decoder_name] = distance_recorder.get_results()
        handler.print('{}-{} distances: {}'.format(args['Identifier'], decoder_name, distance_recorder.get_distances()))
        handler.print('{} aligned tokens out of {} total tokens (in reference)'.format(
            distance_recorder.get_aligned(), distance_recorder.get_true()[0]))

    results['LogProposal'] = logprob_recorder.get_results()
    handler.print('Log Prob (z | x) per token: {:.4f}'.format(logprob_recorder.avg_proposal))

    return results
