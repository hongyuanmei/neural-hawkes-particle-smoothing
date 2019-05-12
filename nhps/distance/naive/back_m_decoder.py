import numpy as np

from nhps.distance.naive.decoder import Decoder
from nhps.distance.utils import DistanceMatrix


class BackMDecoder(Decoder):
    def __init__(self, del_cost, n_types, trans_cost=1.0):
        """

        :param np.ndarray del_cost: A series of floats.
        :param float trans_cost:
        :param int n_types: Not include BOS, EOS and PAD.
        """
        super().__init__(del_cost, trans_cost, n_types)
        self.ks = list()

    @staticmethod
    def _get_mbr_particle(weights, cost_idx, dis_mat, k):
        """
        :param np.ndarray weights:
        :param int cost_idx:
        :param DistanceMatrix dis_mat:
        :param int k:
        :return:
        """
        sub_mat = dis_mat[:k, :k, cost_idx]
        sub_weights = weights[:k]
        scores = (sub_mat * sub_weights).sum(axis=1)
        return int(np.argmin(scores))

    @staticmethod
    def _back_m_helper(weights, cost_idx, dis_mat, k):
        """
        :param np.ndarray weights:
        :param int cost_idx:
        :param DistanceMatrix dis_mat:
        :param int k:
        :return:
        """
        best_idx = BackMDecoder._get_mbr_particle(weights, cost_idx, dis_mat, k)

        epsilon = -1e-3

        # for condition 1, which is related to particles indexed from 1 to K (for code, 0 to k)
        c1_left = (dis_mat.mat[:k, :k, cost_idx] - dis_mat[:k, best_idx, cost_idx]) * weights[:k]
        c1_left = c1_left.sum(axis=1)
        c1_right = weights[k:].sum() * dis_mat[best_idx, :k, cost_idx]
        c1 = c1_left - c1_right
        c1 = all(c1 > epsilon)
        if not c1:
            return best_idx, 1

        # for condition 2, which is related to particles indexed from K+1 to M (for code, k to m)
        c2 = (weights / weights.sum())[best_idx] > 0.5
        if not c2:
            # for condition 3, which is related to particles indexed from K+1 to M (for code, k to m)
            c3_left = weights[best_idx] - weights[k:].sum()
            c3_left = c3_left * dis_mat[best_idx, k:, cost_idx]
            c3_right = (weights[:k] * dis_mat[:k, best_idx, cost_idx]).sum()
            c3 = c3_left - c3_right
            if all(c3 > epsilon):
                return best_idx, 3

        return best_idx, 0

    def _back_m(self, particles, weights):
        """
        :param particles:
        :param weights:
        :return:
        """
        dis_mat = DistanceMatrix(self.compute_particle_distance, particles, self.n_cost)
        rst_indexes = list()

        k_slice = 10
        k_step = int(len(particles) / k_slice)
        ks = list(range(k_step, len(particles), k_step))
        ks.append(len(particles))
        if ks[0] > 10:
            ks[0] = 10

        k = -1
        for cost_idx in range(self.n_cost):
            best_idx = -1
            for k in ks:
                best_idx, condition_satisfied = \
                    self._back_m_helper(weights, cost_idx, dis_mat, k)
                if condition_satisfied == 0:
                    break
            assert best_idx != -1
            rst_indexes.append(best_idx)
        assert k > -1
        return rst_indexes, k, dis_mat

    def decode(self, particles, weights, retain_risk=False):
        """
        :param list particles:
        :param np.ndarray weights:
        :param bool retain_risk:
        """
        particles, weights = self.sort_particles(particles, weights)
        selected_indices, k, dis_mat = self._back_m(particles, weights)
        selected_particles = list()
        for particle_idx in selected_indices:
            selected_particles.append(particles[particle_idx])
        self.ks.append(k)

        if retain_risk:
            risks = np.empty(shape=[self.n_cost], dtype=np.float32)
            for cost_idx in range(self.n_cost):
                distances = dis_mat[selected_indices[cost_idx], :, cost_idx]
                risks[cost_idx] = (distances * weights).sum()
            self.all_risks.append(risks)

        return selected_particles

    def ave_m_prime(self):
        return np.average(self.ks)
