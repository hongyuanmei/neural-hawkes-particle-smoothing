import numpy as np

from nhps.distance.utils import edit_distance_mt_mc


class LogProbRecorder(object):
    def __init__(self, all_log_proposals, all_num_unobs):
        self.all_log_proposals = all_log_proposals
        self.all_num_unobs = all_num_unobs
        total_log_proposal, total_num_unobs = 0.0, 0.0
        all_avg_proposals = []
        for log_proposal, num_unobs in zip(self.all_log_proposals, self.all_num_unobs):
            total_log_proposal += log_proposal
            total_num_unobs += num_unobs
            if num_unobs > 0.5:
                all_avg_proposals.append(log_proposal / num_unobs)
        self.all_avg_proposals = np.array(all_avg_proposals, dtype=np.float32)
        self.avg_proposal = total_log_proposal / total_num_unobs

    def get_results(self):
        rst = {
            'all_avg_proposals': self.all_avg_proposals,
            'avg_proposal': self.avg_proposal
        }
        return rst


class DistanceRecorder(object):
    def __init__(self, del_cost, n_types, trans_cost=1.0):
        """
        :param np.ndarray del_cost:
        :param float trans_cost:
        """
        self.n_cost = len(del_cost)
        self.del_cost_values = del_cost
        self.n_types = n_types
        self.trans_cost = trans_cost

        self.distances = np.zeros(shape=[self.n_cost], dtype=np.float32)
        self.total_trans_cost = np.zeros(shape=[self.n_cost], dtype=np.float32)
        self.num_true = np.zeros(shape=[self.n_cost], dtype=np.int32)
        self.num_del = np.zeros(shape=[self.n_cost], dtype=np.int32)
        self.num_ins = np.zeros(shape=[self.n_cost], dtype=np.int32)
        self.num_align = np.zeros(shape=[self.n_cost], dtype=np.int32)

        self.n_test = 0

    def reset(self):
        self.distances = np.zeros(shape=[self.n_cost], dtype=np.float32)
        self.total_trans_cost = np.zeros(shape=[self.n_cost], dtype=np.float32)
        self.num_true = np.zeros(shape=[self.n_cost], dtype=np.int32)
        self.num_del = np.zeros(shape=[self.n_cost], dtype=np.int32)
        self.num_ins = np.zeros(shape=[self.n_cost], dtype=np.int32)
        self.num_align = np.zeros(shape=[self.n_cost], dtype=np.int32)

        self.n_test = 0

    def record(self, ref, out):
        """

        :param list ref: A ref seq.
        :param list out: A bunch of output seqs.
        """
        self.n_test += 1

        assert len(out) == self.n_cost
        for cost_idx in range(self.n_cost):
            distance_per, trans_cost_per, num_true_per, num_del_per, num_ins_per, num_align_per\
                = edit_distance_mt_mc(ref, out[cost_idx],
                                      self.del_cost_values[cost_idx:cost_idx+1],
                                      self.trans_cost, self.n_types)
            self.distances[cost_idx] += distance_per[0]
            self.total_trans_cost[cost_idx] += trans_cost_per[0]
            self.num_true[cost_idx] += num_true_per[0]
            self.num_del[cost_idx] += num_del_per[0]
            self.num_ins[cost_idx] += num_ins_per[0]
            self.num_align[cost_idx] += num_align_per[0]

    def get_distances(self):
        return self.distances / self.n_test

    def get_aligned(self):
        return self.num_align

    def get_true(self):
        return self.num_true

    def get_results(self):
        jaccard = (self.num_del + self.num_ins).astype(np.float32) / \
                  (self.num_ins + self.num_true)
        insdel = (self.num_del + self.num_ins).astype(np.float32)
        insdel_rate = (self.num_del + self.num_ins).astype(np.float32) / self.num_true
        one_minus_f = (self.num_del + self.num_ins).astype(np.float32) / \
                      (2*self.num_true - self.num_del + self.num_ins)

        cost_per_align = self.total_trans_cost / self.num_align
        cost_per_align[self.num_align == 0] = 0.0

        rst = {
            'costs': self.del_cost_values,
            'distances': self.distances / self.n_test,
            'jaccard': jaccard,
            'insdel': insdel,
            'insdelrate': insdel_rate,
            'oneminusF': one_minus_f,
            'transport': cost_per_align,
            'total_transport': self.total_trans_cost,
            'num_true': self.num_true,
            'num_del': self.num_del,
            'num_ins': self.num_ins,
            'num_align': self.num_align,
            'insdel_per_true': insdel / self.num_true,
            'total_transport_per_true': self.total_trans_cost / self.num_true
        }

        return rst


# Following codes are for testing


if __name__ == '__main__':
    "used for test"
    import pickle
    dataset = pickle.load(open('data/pilotelevator/train.pkl', 'rb'))
    seqs = dataset['seqs']
    #n_types = dataset['obs_num'] + dataset['unobs_num']
    n_types = dataset['total_num']
    from nhpf.eval.distance import ED
    ed = ED(eval_types=range(n_types), cost=0.1, multi=10, num_cost=2)
    print(ed.costs)
    rec = DistanceRecorder(np.array([0.1, 1.0]), n_types)
    seq_idx = 0
    n_test = 10
    for _ in range(n_test):
        ref = seqs[seq_idx]
        out = [seqs[seq_idx+1], seqs[seq_idx+2]]
        ed.compute(out[0], ref, True, 0)
        ed.compute(out[1], ref, True, 1)
        rec.record(ref, out)
        seq_idx += 3
    dis = ed.getDistances()
    dis /= n_test
    print(ed.getResults())
    print(rec.get_results())
    # passed!
