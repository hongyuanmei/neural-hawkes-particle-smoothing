# -*- coding: utf-8 -*-
"""

Paired permutation test

@author: hongyuan
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class PairPerm(object):
    def __init__(self):
        """
        Paired Permutation Test for statistical significance
        """

    def run(self, x, y, num_samples=100000):
        """
        :param x: a list of results provided by model 1
        :param y: a list of results provided by model 2
        :param num_samples: # of boostrapped sets
        :return p_value: < 0.05 means significant with confidence level 95%
        """
        x, y = map(np.array, [x, y])
        d = x - y
        dim = len(d)
        m0 = np.mean(d)
        permutation = (np.random.binomial(1, .5, (num_samples, dim)) * 2 - 1) * d
        mean_perm = np.mean(permutation, 1)
        return float(sum(abs(mean_perm) >= abs(m0))) / mean_perm.shape[0]


if __name__ == '__main__':

    print("generating data ...")
    x = np.random.normal(loc = 0.0, size=1000)
    y = np.random.normal(loc = 0.15, size=1000)

    print("init Paired Permutation Test")
    pp = PairPerm()

    print("run test ...")
    p_value = pp.run(list(x), list(y))

    print("p value : {:.6f}".format(p_value))
