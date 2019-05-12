import numpy as np
import warnings


def remove_base(seq, base, tolerance=1e-4):
    """
    Functionality: Remove x from (x \sqcup z)
    Since there might be some float errors, I allow for a mismatch of the time_stamps between
    two seqs no larger than a threshold.
    The threshold value: tolerance * max_time_stamp
    :param list seq: x \sqcup z
    :param list base: x
    :param float tolerance: A rate.
    :rtype: list
    :return: z
    """
    if len(seq) == 0:
        return seq
    tolerance = tolerance * seq[-1]['time_since_start']
    n_seq = len(seq)
    n_base = len(base)
    seq_types = np.empty(shape=[n_seq], dtype=np.int64)
    seq_time_stamps = np.empty(shape=[n_seq], dtype=np.float32)
    base_types = np.empty(shape=[n_base], dtype=np.int64)
    base_time_stamps = np.empty(shape=[n_base], dtype=np.float32)
    for token_idx, token in enumerate(seq):
        seq_types[token_idx] = token['type_event']
        seq_time_stamps[token_idx] = token['time_since_start']
    for token_idx, token in enumerate(base):
        base_types[token_idx] = token['type_event']
        base_time_stamps[token_idx] = token['time_since_start']

    type_equal = base_types.repeat(n_seq).reshape(n_base, n_seq)
    type_equal = type_equal == seq_types

    time_equal = base_time_stamps.repeat(n_seq).reshape(n_base, n_seq)
    time_equal = np.abs(time_equal - seq_time_stamps) < tolerance

    to_remove = (type_equal & time_equal).any(axis=0)

    rst = list()
    for token_idx in np.where(~to_remove)[0]:
        rst.append(seq[token_idx])

    if len(rst) + len(base) != len(seq):
        warnings.warn('Some base tokens are missing from the seq!')

    return rst


def remove_bases_for_test(all_particles, golds, bases):
    """
    Helper function for testing.
    Functionality: Remove observed tokens from proposed particles and gold seqs.
    :param list all_particles: x \sqcup z_m
    :param list golds: x \sqcup z
    :param list bases: x
    :rtype: list, list
    :return: particles (only z_m) and gold seqs (only z)
    """
    assert len(all_particles) == len(golds) == len(bases)

    rst_particles = list()
    rst_golds = list()

    for particles, gold, base in zip(all_particles, golds, bases):
        new_particles = list()
        for particle in particles:
            new_particles.append(remove_base(particle, base))
        rst_particles.append(new_particles)
        rst_golds.append(remove_base(gold, base))

    return rst_particles, rst_golds


# Following codes are just for testing


if __name__ == '__main__':
    import pickle
    dataset = pickle.load(open('data/pilottaxi/train.pkl', 'rb'))
    seq = dataset['seqs'][0]
    # base = dataset['seqs_obs'][0]
    base = list()
    from pprint import pprint
    pprint('seq:')
    pprint(seq)
    pprint('base:')
    pprint(base)
    pprint('after removal:')
    pprint(remove_base(seq, base))
    assert len(seq) == len(remove_base(seq, base))
