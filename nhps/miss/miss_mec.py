import torch


class MissMec(object):
    def __init__(self, k, device=None):
        """
        :param int k: # of types.
        :param device:
        """
        device = device or 'cpu'
        self.device = torch.device(device)
        self.k = k

    def sample_particles(self, n, event_types=None, time_stamps=None):
        """
        This function should return n observe_mask along with their probabilities (in log space).
        Note: Do NOT mix BOS, EOS, PAD in event seq.
        :param int n: The # of particles required.
        :param torch.Tensor time_stamps: Time stamps. Optional.
        :param torch.Tensor event_types: Event types. Optional.
        :return: 1. The mask for each particle (0 for unobserved and 1 for observed), shaped as [n, m],
        where m is the length of seq.
        2. The log censor_probability for each mask. Shaped as [n]
        """
        raise NotImplementedError

    def compute_probability(self, mask, event_types=None, time_stamps=None):
        """
        This function accepts a seq and its observing mask, return its log censor probability.
        :param torch.Tensor mask: shape=[n, m]
        :param torch.Tensor time_stamps: shape=[n, m]
        :param torch.Tensor event_types: shape=[n, m]
        :return: Log censor probability. shape=[n]
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def neglect_mask(self):
        """
        This function returns a mask indicating which types can be ignored during inference.
        This mask should be shaped as [k], where k is the # of types.
        Elements: 0 for neglectable events, 1 otherwise.
        By default, nothing is neglected.
        :return: Neglect mask.
        """
        return torch.ones(size=[self.k], dtype=torch.float32, device=self.device)

    def r_factor(self, history):
        """
        This function return a vector of length K, where K is the number of event types.
        If possible, each element in the returned vector represents the probability that next proposed event
        is of type k.
        This is possible if it only conditioned on the past history
        (the events observed and proposed before next event).
        If not possible, return all-one vector instead.
        :param history: Proposed events.
        :return: R vector.
        """
        return torch.ones(size=[self.k], dtype=torch.float32, device=self.device)

    def clip_probability(self, idx_from, idx_to, obs_mask, event_types=None, time_stamps=None):
        """
        If the probability could be factorized, you could call this function to compute
        missing probability of a series of continuous events.
        E.g., suppose there is a sequence of events:
        >>> [3, 2, 3, 1, 4]
        If the missing probability could be factorized, you could use this method
        to compute the miss probability from the second event 2 to the fourth event 1.
        :param torch.Tensor idx_from: Beginning indices. Included. Should be an 1d tensor.
        :param torch.Tensor idx_to: Ending indices. Excluded. Should be an 1d tensor.
        :param torch.Tensor event_types: event types. Should be an 2d tensor.
        :param torch.Tensor time_stamps: time stamps. Should be an 2d tensor.
        :rtype: torch.Tensor
        :return a 1d tensor.
        """
        raise NotImplementedError
