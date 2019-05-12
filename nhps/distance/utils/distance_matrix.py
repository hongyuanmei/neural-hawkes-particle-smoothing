import numpy as np


class DistanceMatrix(object):
    def __init__(self, edit_distance, particles, n_cost):
        """
        :param func edit_distance:
        :param list particles:
        :param int n_cost:
        """
        self.calculator = edit_distance
        self.particles = particles
        self.m = len(particles)
        self.n_cost = n_cost
        self.mat = np.empty([self.m, self.m, self.n_cost], dtype=np.float32)
        # Use -2 to indicate a missing value
        self.mat.fill(-2.0)
        self.mat[range(self.m), range(self.m), :] = 0.0

    def fill_cell(self, i, j, k):
        single_cell = isinstance(k, int)
        if isinstance(k, slice):
            start = k.start or 0
            stop = k.stop or self.n_cost
            k = range(start, stop)
        distance = self.calculator(self.particles[i],
                                   self.particles[j],
                                   k)[0]
        if single_cell:
            distance = distance[0]
        self.mat[i, j, k] = distance
        self.mat[j, i, k] = distance

    def __getitem__(self, item):
        assert isinstance(item, tuple)
        assert len(item) == 3
        rst = self.mat[item]
        if (rst < -1).any():

            def convert_slice(index, axis):
                if isinstance(index, int):
                    return range(index, index+1)
                elif isinstance(index, slice):
                    start = index.start or 0
                    stop = index.stop or self.mat.shape[axis]
                    return range(start, stop)

            i_range, j_range = convert_slice(item[0], 0), convert_slice(item[1], 1)

            for i in i_range:
                for j in j_range:
                    if (self.mat[i, j] < -1).any():
                        self.fill_cell(i, j, item[2])
            rst = self.mat[item]

        return rst
