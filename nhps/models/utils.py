import torch
import torch.nn.functional as F


def makeHiddenForBound(linear_weight, states, dtime, particle_num, hidden_dim, total_num):

    c, cb, d, o = states

    c = c.unsqueeze(dim=2).expand(
        particle_num, hidden_dim, total_num)
    cb = cb.unsqueeze(dim=2).expand_as(c)
    d = d.unsqueeze(dim=2).expand_as(c)
    o = o.unsqueeze(dim=2).expand_as(c)

    dtime = dtime.unsqueeze(dim=1).unsqueeze(dim=2).expand_as(c)
    linear_weight = linear_weight.unsqueeze(dim=0).expand_as(c)

    cgap = c - cb

    indices_inc_0 = (cgap > 0.0) & (linear_weight < 0.0)
    indices_inc_1 = (cgap < 0.0) & (linear_weight > 0.0)

    cgap[indices_inc_0] = 0.0
    cgap[indices_inc_1] = 0.0

    cy = cb + cgap * torch.exp(-d * dtime)
    hy = o * torch.tanh(cy)

    return hy


def makeHiddenForLambda(states, dtime):
    c, cb, d, o = states
    # particle_num * hidden_dim
    # shape of dtime : particle_num * num(=100)
    c = c.unsqueeze(dim=1)
    cb = cb.unsqueeze(dim=1)
    d = d.unsqueeze(dim=1)
    o = o.unsqueeze(dim=1)
    dtime = dtime.unsqueeze(dim=2)

    cy = cb + (c - cb) * torch.exp(-d * dtime)
    hy = o * torch.tanh(cy)

    return hy


def combine_dict(*dicts):
    """
    :param dict dicts:
    :rtype: dict
    """
    rst = dict()
    for single_dict in dicts:
        for key, value in single_dict.items():
            assert key not in rst
            rst[key] = value
    return rst
