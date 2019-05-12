import numpy as np

from nhps.distance.utils.edit_distance import edit_distance_mt_mc


class Decoder(object):
    """
    Abstract class
    """
    def __init__(self, del_cost, trans_cost, n_types):
        """

        :param np.ndarray del_cost: A series of floats.
        :param float trans_cost:
        :param int n_types: Not include BOS, EOS and PAD.
        """
        self.n_cost = len(del_cost)
        self.del_cost = del_cost
        self.trans_cost = trans_cost
        self.n_types = n_types

        self.all_risks = list()

    def decode(self, particles, weights, retain_risk=False):
        """

        :param list particles: Particles.
        :param np.ndarray weights: Corresponding weights.
        :param bool retain_risk:
        :return: Decoded particle.
        """
        raise NotImplementedError

    def compute_particle_distance(self, seq1, seq2, del_cost_idx=None):
        """

        :param list seq1:
        :param list seq2:
        :param del_cost_idx:
        """
        del_cost_idx = del_cost_idx or range(len(self.del_cost))
        if isinstance(del_cost_idx, int):
            del_cost_idx = range(del_cost_idx, del_cost_idx+1)
        del_cost_values = self.del_cost[del_cost_idx]
        return edit_distance_mt_mc(seq1, seq2, del_cost_values, self.trans_cost, self.n_types)

    @staticmethod
    def sort_particles(particles, weights):
        """

        :param np.ndarray weights:
        :param list particles:
        """
        orders = np.argsort(-weights)
        new_weights = weights[orders]
        new_particles = list()
        for particle_idx in orders:
            new_particles.append(particles[particle_idx])
        return new_particles, new_weights

    def ave_risk(self):
        """
        For stats.
        Compute the average risk for each del_cost.
        :return:
        """
        risks = np.array(self.all_risks)
        return np.average(risks, axis=0)
