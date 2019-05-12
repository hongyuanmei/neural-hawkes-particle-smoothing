import numpy as np

from nhps.distance.utils import max_triangle_1d, float_equal


class Token(object):
    def __init__(self, time_stamp, del_cost):
        self.time_stamp = time_stamp
        self.cost = del_cost
        self.chords = list()

    def re_adjust_positions(self, particles, del_cost, trans_cost):
        n_chords = len(self.chords)
        if n_chords == 0:
            return False
        weights = np.empty(shape=[n_chords], dtype=np.float32)
        time_stamps = np.empty(shape=[n_chords], dtype=np.float32)
        for chord_idx, [particle_idx, particle_token_idx] in enumerate(self.chords):
            particle = particles[particle_idx]
            weights[chord_idx] = particle.weight
            time_stamps[chord_idx] = particle.time_stamps[particle_token_idx]
        new_time_stamp_idx, max_saved = max_triangle_1d(2*del_cost/trans_cost,
                                                        time_stamps, weights*del_cost*2)
        new_time_stamp = time_stamps[new_time_stamp_idx]
        curr_cost = del_cost - max_saved
        self.cost = curr_cost
        modified = not float_equal(self.time_stamp, new_time_stamp)
        self.time_stamp = new_time_stamp
        return modified

    def reset(self, del_cost):
        self.cost = del_cost
        self.chords = list()

    def recalculate_cost(self, particles, del_cost, trans_cost):
        """
        Might be costly. Used only for debugging.
        :param list particles:
        :param float del_cost:
        :param float trans_cost:
        :rtype: float
        """
        cost = del_cost
        for particle_idx, particle_token_idx in self.chords:
            particle = particles[particle_idx]
            distance = abs(particle.time_stamps[particle_token_idx]-self.time_stamp)
            saved_dis = 2 * del_cost / trans_cost - distance
            if saved_dis < 0:
                continue
            cost -= saved_dis * trans_cost * particle.weight
        return cost
