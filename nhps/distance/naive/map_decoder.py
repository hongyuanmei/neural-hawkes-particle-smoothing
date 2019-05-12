import numpy as np

from nhps.distance.naive.decoder import Decoder
from nhps.distance.utils import DistanceMatrix


class MapDecoder(Decoder):
    def __init__(self, del_cost, n_types, trans_cost=1.0):
        """

        :param np.ndarray del_cost: A series of floats.
        :param float trans_cost:
        :param int n_types: Not include BOS, EOS and PAD.
        """
        super().__init__(del_cost, trans_cost, n_types)

    def decode(self, particles, weights, retain_risk=False):
        """

        :param list particles:
        :param np.ndarray weights:
        :param bool retain_risk:
        :return:
        """
        assert len(particles) == len(weights)
        max_idx = int(np.argmax(weights))

        if retain_risk:
            dis_mat = DistanceMatrix(self.compute_particle_distance, particles, self.n_cost)
            distances = dis_mat[max_idx, :, :].transpose([1, 0])
            self.all_risks.append((distances * weights).sum(axis=1))

        return [particles[max_idx]] * self.n_cost
