import numpy as np


def find_alignment_mc(seq1, seq2, del_cost, trans_cost):
    """
    We use dynamic programming to find the best alignments between two seqs.
    ``nc'' means that this functions support a series of del_cost values.
    Note: Not support multiple types.
    :param np.ndarray seq1: Time stamps of seq #1.
    :param np.ndarray seq2: Time stamps of seq #2.
    :param np.ndarray del_cost: A series of delete cost.
    :param float trans_cost: Transportation cost per unit length.
    :return: Alignment list and minimum distances for all the del_cost values.
    """
    n_cost = len(del_cost)
    n1 = len(seq1)
    n2 = len(seq2)
    # shape=[n2, n1]
    trans_mask = np.abs(seq2.repeat(n1).reshape(n2, n1)-seq1) * trans_cost
    # shape=[n1+1, n1+1]
    del_mask = np.arange(n1+2, dtype=np.float32) \
                   .repeat(n1+1).reshape(n1+2, n1+1) \
                   .T.reshape(-1)[:(n1+1)**2].reshape(n1+1, n1+1) - 1
    del_mask[np.tril_indices(n1+1, -1)] = float('inf')
    # shape=[n1+1, n1+1, n_cost]
    del_mask = del_mask.repeat(n_cost).reshape(n1+1, n1+1, n_cost) * del_cost
    # shape=[n1+1, n1+1, n_cost]
    del_mask = del_mask.transpose([1, 0, 2]).copy()
    # shape=[n1+1, n_cost]
    overhead = np.empty(shape=[n1+1, n_cost], dtype=np.float32)
    overhead.fill(float('inf'))
    overhead[0, :] = 0.0
    # shape=[n2, n1+1, n_cost]
    back_pointers = np.empty(shape=[n2, n1+1, n_cost], dtype=np.int32)
    for n2_idx in range(n2):
        # shape=[n1+1, n1+1, n_cost]
        add_mask = del_mask.copy()
        add_mask[1:, :, :] += np.outer(trans_mask[n2_idx],
                                       np.ones(shape=[(n1+1)*n_cost],
                                               dtype=np.float32)).reshape(n1, n1+1, n_cost)
        add_mask[np.arange(n1+1), np.arange(n1+1), :] = del_cost
        # shape=[n1+1, n1+1, n_cost]
        cost_mat = overhead + add_mask
        # shape=[n1+1, n_cost]
        choices = np.argmin(cost_mat, axis=1)
        back_pointers[n2_idx] = choices
        overhead = cost_mat.min(axis=1)
    overhead += np.outer(np.arange(n1, -1, -1, dtype=np.float32), np.ones(shape=[n_cost])) * del_cost
    # shape=[n_cost]
    curr_choice = np.argmin(overhead, axis=0)
    # shape=[n_cost]
    min_distance = overhead.min(axis=0)
    best_route = [curr_choice]
    # shape=[n1+1, n_cost]
    for choice_list in back_pointers[::-1]:
        # shape=[n_cost]
        curr_choice = choice_list[curr_choice, np.arange(n_cost)]
        best_route.append(curr_choice)
    # shape=[n2, n_cost]
    best_route = np.array(best_route)

    align_pairs = list()
    for cost_idx in range(n_cost):
        best_route_ = best_route[:, cost_idx]
        pairs = list()
        memo = -1
        for n2_idx_plus_1, choice_made in enumerate(best_route_[::-1]):
            if choice_made != memo:
                pairs.append([choice_made-1, n2_idx_plus_1-1])
            memo = choice_made
        align_pairs.append(pairs[1:])

    return [align_pairs, # len=n_cost
            min_distance # shape=[n_cost]
            ]


def find_alignment(seq1, seq2, del_cost, trans_factor):
    """
    Similar functionality with find_alignment_nc, but for single del_cost cost.
    :param np.ndarray seq1:
    :param np.ndarray seq2:
    :param float del_cost:
    :param float trans_factor:
    :return:
    """
    align_pairs, min_distance = \
        find_alignment_mc(seq1, seq2, np.array([del_cost]), trans_factor)
    return align_pairs[0], float(min_distance[0])


def float_equal(a, b):
    eps = 1e-4
    return (1-eps) < (a/b) < (1+eps)


def edit_distance_mt_mc(ref, decoded, del_cost, trans_cost, n_types):
    """

    :param list ref:
    :param list decoded:
    :param np.ndarray del_cost:
    :param float trans_cost:
    :param int n_types:
    """
    num_cost = len(del_cost)

    distances = np.zeros(shape=[num_cost], dtype=np.float32)
    total_trans_cost = np.zeros(shape=[num_cost], dtype=np.float32)
    num_true = np.zeros(shape=[num_cost], dtype=np.int32)
    num_del = np.zeros(shape=[num_cost], dtype=np.int32)
    num_ins = np.zeros(shape=[num_cost], dtype=np.int32)
    num_align = np.zeros(shape=[num_cost], dtype=np.int32)

    seq_per_types = [[list(), list()] for _ in range(n_types)]
    for seq_idx, seq in enumerate([ref, decoded]):
        for token in seq:
            event_type = token['type_event']
            if event_type >= n_types:
                continue
            seq_per_types[event_type][seq_idx].append(token['time_since_start'])

    for type_idx in range(n_types):
        ref_time = np.array(seq_per_types[type_idx][0])
        decoded_time = np.array(seq_per_types[type_idx][1])
        align_pairs, min_distance = find_alignment_mc(
            ref_time, decoded_time, del_cost, trans_cost)
        for cost_idx in range(num_cost):
            align_pairs_per_cost = align_pairs[cost_idx]
            min_distance_per_cost = min_distance[cost_idx]
            num_align[cost_idx] += len(align_pairs_per_cost)
            num_true[cost_idx] += len(ref_time)
            n_ins_per_cost = len(decoded_time) - len(align_pairs_per_cost)
            n_del_per_cost = len(ref_time) - len(align_pairs_per_cost)
            num_ins[cost_idx] += n_ins_per_cost
            num_del[cost_idx] += n_del_per_cost
            distances[cost_idx] += min_distance_per_cost
            total_trans_cost[cost_idx] += min_distance_per_cost\
                                          - del_cost[cost_idx]*(n_ins_per_cost+n_del_per_cost)

    return distances, total_trans_cost, num_true, num_del, num_ins, num_align


# Following codes are for testing


if __name__ == '__main__':
    "used for test"
    import pickle
    seqs = pickle.load(open('data/pilotelevator/dev.pkl', 'rb'))['seqs']
    ref = seqs[10]
    decoded = seqs[11]
    del_cost = np.array([0.1, 100])
    trans_cost = 1.0
    import time
    since = time.time()
    for _ in range(100):
        rst1 = edit_distance_mt_mc(ref, decoded, del_cost, trans_cost, 10)
    print(time.time() - since)
    from nhpf.eval.distance import ED
    ed = ED(eval_types=range(10), cost=0.1, multi=1000, num_cost=2)
    since = time.time()
    for _ in range(100):
        rst2 = ed.compute(decoded, ref)
    print(time.time() - since)
    for i in range(6):
        print(rst1[i])
        print(rst2[i], '\n')
    x = 1
