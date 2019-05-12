import torch
import numpy as np

from .miss_mec import MissMec


class FactorizedMissMec(MissMec):
    def __init__(self, device, config_file, eps=1e-5):
        """
        :param str config_file: configuration file path
        :param device:
        :param float eps:
        """
        with open(config_file, 'r') as fp:
            nums = fp.read().split()
        nums = [float(num) for num in nums]
        # probability of miss
        self.pr = np.array(nums)
        super().__init__(len(self.pr), device)
        self.eps = eps

    def sample_particles(self, n, event_types=None, time_stamps=None):
        assert event_types is not None
        m = len(event_types)
        random_floats = np.random.random([n, m])
        threshold_idx = event_types.cpu().numpy()
        threshold = self.pr[threshold_idx]
        mask = (random_floats > threshold).astype(np.float32).copy()
        miss_pr = self.pr[threshold_idx]
        censor_probs = (1 - 2 * miss_pr) * mask + miss_pr
        censor_probs[censor_probs < self.eps] = self.eps
        log_censor_prob = np.log(censor_probs).sum(axis=1)
        return torch.tensor(mask).to(self.device, dtype=torch.float32),\
               torch.tensor(log_censor_prob).to(self.device, dtype=torch.float32)

    def compute_probability(self, mask, event_types=None, time_stamps=None):
        # shape=[n, m], where n is the # of seqs, and m is the length
        if not isinstance(mask, np.ndarray):
            mask = mask.cpu().numpy()
        event_types = event_types.cpu().numpy()
        oov_mask = event_types >= len(self.pr)
        event_types = event_types.copy()
        event_types[oov_mask] = 0
        miss_pr = self.pr[event_types]
        censor_probs = (1 - 2 * miss_pr) * mask + miss_pr
        censor_probs[oov_mask] = 1.0
        censor_probs[censor_probs < self.eps] = self.eps
        log_censor_prob = np.log(censor_probs).sum(axis=1)
        return torch.tensor(log_censor_prob).to(device=self.device, dtype=torch.float32)

    def neglect_mask(self):
        return torch.tensor((self.pr > self.eps).astype(np.float32)).to(self.device, dtype=torch.float32)

    def r_factor(self, history):
        """
        In the case of factorized case,
        each event is independently censored.
        :param history: not used here.
        :return: R vector.
        """
        return torch.tensor(self.pr).to(device=self.device, dtype=torch.float32)

    def clip_probability(self, idx_from, idx_to, obs_mask, event_types=None, time_stamps=None):
        # time_stamps is not used here.
        idx_from, idx_to = idx_from.cpu().numpy(), idx_to.cpu().numpy()
        event_types = event_types.cpu().numpy()
        n_particle, n_events = event_types.shape
        index_mask_help = np.arange(n_events).repeat(n_particle).reshape(n_events, n_particle).T
        index_mask = np.full(shape=[n_particle, n_events], fill_value=True, dtype=np.bool)
        index_mask = index_mask & (index_mask_help.T >= idx_from).T
        index_mask = index_mask & (index_mask_help.T < idx_to).T

        oov_mask = event_types >= len(self.pr)
        event_types = event_types.copy()
        event_types[oov_mask] = 0
        miss_pr = self.pr[event_types]
        censor_probs = (1 - 2 * miss_pr) * obs_mask.cpu().numpy() + miss_pr
        censor_probs[oov_mask] = 1.0
        censor_probs[~index_mask] = 1.0
        censor_probs[censor_probs < self.eps] = self.eps
        log_censor_prob = np.log(censor_probs).sum(axis=1)

        return torch.tensor(log_censor_prob).to(device=self.device, dtype=torch.float32)
