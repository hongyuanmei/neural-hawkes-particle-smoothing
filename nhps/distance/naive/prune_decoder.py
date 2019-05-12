import numpy as np

from nhps.distance.naive.decoder import Decoder
from nhps.distance.utils.distance_matrix import DistanceMatrix


class PruneDecoder(Decoder):
    def __init__(self, del_cost, n_types, threshold, trans_cost=1.0):
        """

        :param np.ndarray del_cost: A series of floats.
        :param int n_types: Not include BOS, EOS and PAD.
        :param float threshold:
        :param float trans_cost:
        """
        super().__init__(del_cost, trans_cost, n_types)
        self.threshold = threshold
        self.n_lefts = list()

    def prune_particles(self, particles, weights):
        particles, weights = self.sort_particles(particles, weights)
        reduce_sum = np.cumsum(weights[::-1])
        n_left = int((reduce_sum > self.threshold).sum())
        self.n_lefts.append(n_left)
        return particles[:n_left], weights[:n_left]

    def decode(self, particles, weights, retain_risk=False):
        """

        :param list particles:
        :param np.ndarray weights:
        :param bool retain_risk:
        """
        particles, weights = self.prune_particles(particles, weights)
        dis_mat = DistanceMatrix(self.compute_particle_distance, particles,
                                 len(self.del_cost))[:, :, :].transpose([2, 0, 1])
        scores = (dis_mat * weights).sum(axis=2)
        selected_indices = np.argmin(scores, axis=1)
        selected_particles = list()
        for particle_idx in selected_indices:
            selected_particles.append(particles[particle_idx])

        if retain_risk:
            risks = np.empty(shape=[self.n_cost], dtype=np.float32)
            for cost_idx in range(self.n_cost):
                distances = dis_mat[cost_idx, selected_indices[cost_idx], :]
                risks[cost_idx] = (distances * weights).sum()
            self.all_risks.append(risks)

        return selected_particles

    def ave_left(self):
        return np.average(self.n_lefts)
