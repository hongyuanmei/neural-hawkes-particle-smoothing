# -*- coding: utf-8 -*-
"""

Processers

@author: hongyuan
"""

import torch
import numpy as np


def sampleForIntegral(input, sampling=1, device=None):
    r"""
    sampling dtimes in each interval given other tensors
    this function only deals with particles of the same seq
    so their duration is the same, but lens may be different
    """

    r"""
    for particles of the same seq, we should use
    same seq of randomly sampled times to compute integral
    because, otherwise, the same particle may get different weights
    so we always bias to the higher-weighted one
    in this way, we are not maximizing the log-likelihood
    instead, we would be maximizing the random upper reach of the log-likelihood
    and there is no guarantee where the mean would go
    to get rid of this bias, for all particles of each seq,
    we should use the EXACTLY same seq of sampled times

    If in the future we want to compare particles while we sample them
    we can precompute this seq of times and compute the integral along the way
    i.e. interval after interval
    """

    device = device or 'cpu'
    device = torch.device(device)
    event, time, post, duration, lens, \
    event_obs, dtime_obs, dtime_backward, index_of_hidden_backward, \
    mask_obs, mask_unobs, log_censor_probs = input

    num_particles, T_plus_2 = event.size()
    _, T_obs_plus_2 = event_obs.size()
    assert lens.max() + 2 == T_plus_2, "max len should match"

    max_sampling = max( int( lens.max() * sampling ), 1 )

    sampled_times = torch.empty(size=[max_sampling], dtype=torch.float32,
                                device=device)
    sampled_times.uniform_(0.0, float(duration[0]))
    sampled_times = sampled_times.sort()[0].unsqueeze(0).expand(num_particles, max_sampling)

    dtime_sampling = torch.zeros(size=[num_particles, max_sampling],
                                 dtype=torch.float32, device=device)

    index_of_hidden_sampling = torch.zeros(size=[num_particles, max_sampling],
                                    dtype=torch.int64, device=device)

    dtime_backward_sampling = torch.zeros(size=[num_particles, max_sampling],
                                          dtype=torch.float32, device=device)

    index_of_hidden_backward_sampling = torch.zeros(size=[num_particles, max_sampling],
                                            dtype=torch.int64, device=device)

    cum_time = time.cumsum(dim=1)
    indices_mat = torch.arange(0, num_particles, dtype=torch.int64,
                               device=device).unsqueeze(1).expand(num_particles, max_sampling)
    indices_mat = (T_plus_2 - 1) * indices_mat

    current_step = torch.zeros(size=[num_particles, max_sampling],
                               dtype=torch.int64, device=device)

    for j in range( lens.max() + 1 ):

        bench_cum_time = cum_time[:, j].unsqueeze(1).expand(
            num_particles, max_sampling)
        indices_to_edit = sampled_times > bench_cum_time

        dtime_sampling[indices_to_edit] = \
        (sampled_times - bench_cum_time)[indices_to_edit]

        current_step.fill_(j)
        index_of_hidden_sampling[indices_to_edit] = \
        (indices_mat + current_step)[indices_to_edit]

    assert dtime_sampling.min() >= 0.0, "Time >= 0"

    cum_time_obs = dtime_obs.cumsum(dim=1)
    indices_mat_obs = torch.zeros(size=[num_particles, max_sampling],
                                  dtype=torch.int64, device=device)
    current_step_obs = torch.zeros(size=[num_particles, max_sampling],
                                   dtype=torch.int64, device=device)

    for j in range( T_obs_plus_2 - 1 ):

        bench_cum_time = cum_time_obs[:, j].unsqueeze(1).expand(
            num_particles, max_sampling)
        ceiling_cum_time = cum_time_obs[:, j+1].unsqueeze(1).expand(
            num_particles, max_sampling)
        indices_to_edit = (sampled_times > bench_cum_time) & (sampled_times <= ceiling_cum_time)

        dtime_backward_sampling[indices_to_edit] = \
        (ceiling_cum_time - sampled_times)[indices_to_edit]

        current_step_obs.fill_(j)
        index_of_hidden_backward_sampling[indices_to_edit] = \
        (indices_mat_obs + current_step_obs)[indices_to_edit]

    return event, time, post, duration, dtime_sampling, index_of_hidden_sampling, \
    event_obs, dtime_obs, dtime_backward, index_of_hidden_backward, \
    dtime_backward_sampling, index_of_hidden_backward_sampling, \
    mask_obs, mask_unobs, log_censor_probs
    # idx of output :
    # event 0, time 1, post 2, duration 3,
    # dtime_sampling 4, index_of_hidden_sampling 5,
    # event_obs 6, dtime_obs 7,
    # dtime_backward 8, index_of_hidden_backward 9, \
    # dtime_backward_sampling 10, index_of_hidden_backward_sampling 11
    # mask_obs 12, mask_unobs 13, log_censor_probs, 14


def processBatchParticles(
    batch_of_seqs, idx_BOS, idx_EOS, idx_PAD, device=None):

    device = device or 'cpu'
    device = torch.device(device)

    batch_size = len(batch_of_seqs)
    num_particles = batch_of_seqs[0][2].size(0)

    max_len = -1
    max_len_sampling = -1
    max_len_obs = -1

    for i_batch, seq_with_particles in enumerate(batch_of_seqs):
        seq_len = seq_with_particles[0].size(1)
        seq_len_sampling = seq_with_particles[4].size(1)
        seq_len_obs = seq_with_particles[6].size(1)
        max_len = seq_len if seq_len > max_len else max_len
        max_len_sampling = seq_len_sampling if seq_len_sampling > max_len_sampling else max_len_sampling
        max_len_obs = seq_len_obs if seq_len_obs > max_len_obs else max_len_obs

    post = torch.zeros(size=[batch_size, num_particles], dtype=torch.float32,
                         device=device)
    duration = torch.zeros(size=[batch_size, num_particles], dtype=torch.float32, device=device)

    r"""
    modify all the vocab size to the right idx : idx_BOS, idx_EOS, idx_EOS
    """
    event = torch.empty(size=[batch_size, num_particles, max_len],
                        dtype=torch.int64, device=device).fill_(idx_PAD)
    time = torch.zeros(size=[batch_size, num_particles, max_len],
                       dtype=torch.float32, device=device)

    dtime_sampling = torch.zeros(size=[batch_size, num_particles, max_len_sampling],
                                 dtype=torch.float32, device=device)
    index_of_hidden_sampling = torch.zeros(size=[batch_size, num_particles, max_len_sampling],
                                           dtype=torch.int64, device=device)
    mask_sampling = torch.zeros(size=[batch_size, num_particles, max_len_sampling],
                                dtype=torch.float32, device=device)
    # note we use batch_size as 0-dim here
    # because we need to flatten num_particles and max_len_sampling
    # in forward method of nhp

    event_obs = torch.empty(size=[batch_size, num_particles, max_len_obs],
                            dtype=torch.int64, device=device).fill_(idx_PAD)
    time_obs = torch.zeros(size=[batch_size, num_particles, max_len_obs],
                           dtype=torch.float32, device=device)

    dtime_backward = torch.zeros(size=[batch_size, num_particles, max_len],
                                 dtype=torch.float32, device=device)
    index_of_hidden_backward = torch.zeros(size=[batch_size, num_particles, max_len],
                                           dtype=torch.int64, device=device)

    dtime_backward_sampling = torch.zeros(size=[batch_size, num_particles, max_len_sampling],
                                          dtype=torch.float32, device=device)
    index_of_hidden_backward_sampling = torch.zeros(
        size=[batch_size, num_particles, max_len_sampling],
        dtype=torch.int64,
        device=device
    )

    mask_obs = torch.zeros( size=[batch_size, num_particles, max_len],
        dtype=torch.float32, device=device )
    mask_unobs = torch.zeros( size=[batch_size, num_particles, max_len],
        dtype=torch.float32, device=device )
    log_censor_probs = torch.zeros( size=[batch_size, num_particles],
        dtype=torch.float32, device=device )

    for i_batch, seq_with_particles in enumerate(batch_of_seqs):
        seq_len = seq_with_particles[0].size(1)
        seq_len_sampling = seq_with_particles[4].size(1)
        seq_len_obs = seq_with_particles[6].size(1)

        event[i_batch, :, :seq_len] = seq_with_particles[0].clone()
        time[i_batch, :, :seq_len] = seq_with_particles[1].clone()

        post[i_batch, :] = seq_with_particles[2].clone()
        duration[i_batch, :] = seq_with_particles[3].clone()

        dtime_sampling[i_batch, :, :seq_len_sampling] = seq_with_particles[4].clone()
        mask_sampling[i_batch, :, :seq_len_sampling] = 1.0

        event_obs[i_batch, :, :seq_len_obs] = seq_with_particles[6].clone()
        time_obs[i_batch, :, :seq_len_obs] = seq_with_particles[7].clone()

        dtime_backward[i_batch, :, :seq_len] = seq_with_particles[8].clone()
        dtime_backward_sampling[i_batch, :, :seq_len_sampling] = \
        seq_with_particles[10].clone()

        r"""
        since we now have an extra dimension i.e. batch_size
        we need to revise the index_of_hidden_sampling, that is,
        it should not be i_particle * (T+1) + j anymore
        what it should be ?
        consider when we flat the states, we make them to
        ( batch_size * num_particles * T+1 ) * hidden_dim
        and when we flatten the index_of_hidden_sampling, it is
        batch_size * num_particles * max_len_sampling
        so each entry should be :
        i_seq * ( num_particles * (T+1) ) + i_particle * (T+1) + j , that is
        for whatever value of element in this current matrix
        we should add it with i_seq * ( num_particles * (T+1) )
        this part is tricky
        """
        remainder = seq_with_particles[5] % ( seq_len - 1 )
        multiple = seq_with_particles[5] / ( seq_len - 1 )
        index_of_hidden_sampling[i_batch, :, :seq_len_sampling] = \
        i_batch * num_particles * (max_len - 1) + multiple * (max_len - 1) + remainder

        remainder = seq_with_particles[9] % ( seq_len_obs - 1 )
        multiple = seq_with_particles[9] / ( seq_len_obs - 1 )
        index_of_hidden_backward[i_batch, :, :seq_len] = \
        i_batch * num_particles * (max_len_obs - 1) + multiple * (max_len_obs - 1) + remainder

        remainder = seq_with_particles[11] % ( seq_len_obs - 1 )
        multiple = seq_with_particles[11] / ( seq_len_obs - 1 )
        index_of_hidden_backward_sampling[i_batch, :, :seq_len_sampling] = \
        i_batch * num_particles * (max_len_obs - 1) + multiple * (max_len_obs - 1) + remainder

        mask_obs[i_batch, :, :seq_len] = seq_with_particles[12].clone()
        mask_unobs[i_batch, :, :seq_len] = seq_with_particles[13].clone()
        log_censor_probs[i_batch, :] = seq_with_particles[14].clone()

    return [
        event,
        time,
        post,
        duration,
        dtime_sampling,
        index_of_hidden_sampling,
        mask_sampling,
        event_obs,
        time_obs,
        dtime_backward,
        index_of_hidden_backward,
        dtime_backward_sampling,
        index_of_hidden_backward_sampling,
        mask_obs,
        mask_unobs,
        log_censor_probs
    ]


class DataProcessorBase(object):
    def __init__(self, idx_BOS, idx_EOS, idx_PAD, miss_mec, sampling=1, device=None):
        self.idx_BOS = idx_BOS
        self.idx_EOS = idx_EOS
        self.idx_PAD = idx_PAD
        self.sampling = sampling
        self.miss_mec = miss_mec

        device = device or 'cpu'
        self.device = torch.device(device)

        self.funcBatch = processBatchParticles
        self.sampleForIntegral = sampleForIntegral

    def getSeq(self, event, dtime, len_seq):

        dtime_cum = dtime.cumsum(dim=0)

        assert event[0] == self.idx_BOS, "starting with BOS"
        assert event[len_seq + 1] == self.idx_EOS, "ending with EOS"

        out = []
        for i in range(len_seq):
            out.append(
                {
                    'type_event': int(event[i + 1]),
                    'time_since_last_event': float(dtime[i + 1]),
                    'time_since_start': float(dtime_cum[i + 1])
                }
            )
        # out does NOT include BOS or EOS
        return out

    def orgSeq(self, seq, duration=None):
        """
        :param list seq:
        :param float duration:
        :return:
        """
        if len(seq) < 1:
            assert duration is not None, "duration is needed for empty seq"
        duration = duration or seq[-1]['time_since_start']
        new_seq = seq.copy()
        bos_token = {
            'type_event': self.idx_BOS, 'time_since_last_event': 0.0,
            'time_since_start': 0.0
        }
        if len(seq) > 0:
            last_event_time = seq[-1]['time_since_start']
        else:
            last_event_time = 0
        eos_token = {
            'type_event': self.idx_EOS, 'time_since_last_event': duration-last_event_time,
            'time_since_start': duration
        }
        new_seq.insert(0, bos_token)
        new_seq.append(eos_token)
        return new_seq

    def processSeq(self, seq, n=1, seq_obs=None):
        """
        The process seq function is moved to the class.
        :param list seq:
        :param int n: # of obs_seqs
        """
        # including BOS and EOS
        len_seq = len(seq)+2
        event = torch.zeros(size=[len_seq], device=self.device, dtype=torch.int64)
        event[0], event[-1] = self.idx_BOS, self.idx_EOS
        dtime = torch.zeros(size=[len_seq], device=self.device, dtype=torch.float32)
        for token_idx, token in enumerate(seq):
            event[token_idx+1] = int(token['type_event'])
            dtime[token_idx+1] = float(token['time_since_last_event'])
        time_stamps = dtime.cumsum(dim=0)
        duration = torch.empty(size=[n], dtype=torch.float32, device=self.device).fill_(time_stamps[-1])
        posterior = torch.zeros(size=[n], dtype=torch.float32, device=self.device)
        lens = torch.empty(size=[n], dtype=torch.int64, device=self.device).fill_(len_seq-2)

        # mask: [n, m], log_censor_prob: [n]
        if seq_obs is None:
            masks, log_censor_probs = self.miss_mec.sample_particles(
                n, event[1:-1], time_stamps[1:-1])
        else:
            assert n == 1, "one seq obs is given, why partition is not 1?"
            masks = torch.zeros(size=[n, len_seq - 2], dtype=torch.float32, device=self.device)
            idx_in_complete = 0
            for idx_in_obs, token_obs in enumerate(seq_obs):
                event_type_obs = seq_obs[idx_in_obs]['type_event']
                time_since_start_obs = seq_obs[idx_in_obs]['time_since_start']

                def check_type_and_time(idx_in_complete):
                    event_type_complete = seq[idx_in_complete]['type_event']
                    time_since_start_complete = seq[idx_in_complete]['time_since_start']
                    event_type_equal = event_type_complete == event_type_obs
                    eps = 1e-5
                    time_equal = ( time_since_start_complete == time_since_start_obs == 0) or (1-eps) < time_since_start_complete / time_since_start_obs < (1+eps)
                    return event_type_equal and time_equal

                while idx_in_complete < len(seq) and not check_type_and_time(idx_in_complete):
                    idx_in_complete += 1

                assert idx_in_complete < len(seq), "Current obs token not in complete?"

                masks[0, idx_in_complete] = 1.0

            log_censor_probs = self.miss_mec.compute_probability(
                masks, event_types=event[1:-1].unsqueeze(0), time_stamps=time_stamps[1:-1].unsqueeze(0))

        masks_np = masks.detach().cpu().numpy()
        longest_obs = int((masks_np > 0.5).sum(axis=1).max()) + 2

        event_obs = torch.empty(size=[n, longest_obs], dtype=torch.int64, device=self.device).fill_(self.idx_PAD)
        dtime_obs = torch.zeros(size=[n, longest_obs], dtype=torch.float32, device=self.device)
        dtime_backward = torch.zeros(size=[n, len_seq], dtype=torch.float32, device=self.device)
        index_of_hidden_backward = torch.zeros(size=[n, len_seq], dtype=torch.int64, device=self.device)

        for mask_idx, mask in enumerate(masks):
            obs_indexes = torch.arange(len(mask), dtype=torch.int64, device=self.device)[mask > 0.5] + 1
            obs_indexes = obs_indexes.cpu().tolist()
            obs_indexes.insert(0, 0)
            obs_indexes.append(len_seq-1)

            last_time_stamp = 0.0
            for obs_idx, token_idx in enumerate(obs_indexes):
                event_obs[mask_idx, obs_idx] = event[token_idx]
                dtime_obs[mask_idx, obs_idx] = time_stamps[token_idx] - last_time_stamp
                last_time_stamp = time_stamps[token_idx]

            obs_idx = 0
            # Use indexing instead of cumsum in order to prevent float error.
            obs_time_stamps = time_stamps[obs_indexes]
            for token_idx in range(len_seq):
                while obs_indexes[obs_idx] < token_idx:
                    obs_idx += 1

                dtime_backward[mask_idx, token_idx] = obs_time_stamps[obs_idx] - time_stamps[token_idx]
                index_of_hidden_backward[mask_idx, token_idx] = mask_idx * (longest_obs-1) + obs_idx - 1

        event = event.repeat(n).reshape(n, len_seq)
        dtime = dtime.repeat(n).reshape(n, len_seq)

        mask_obs = torch.cat([torch.zeros(size=[n, 1], dtype=torch.float32, device=self.device),
                              masks,
                              torch.zeros(size=[n, 1], dtype=torch.float32, device=self.device)],
                             dim=1)

        mask_unobs = torch.cat([torch.zeros(size=[n, 1], dtype=torch.float32, device=self.device),
                                1-masks,
                                torch.zeros(size=[n, 1], dtype=torch.float32, device=self.device)],
                               dim=1)

        return [event, # 0: n * len_seq
                dtime, # 1: n * len_seq
                posterior, # 2: n
                duration, # 3: n
                lens, # 4: n # total len without BOS or EOS
                event_obs, # 5: n * len_seq_obs (with PAD)
                dtime_obs, # 6: n * len_seq_obs (with PAD)
                dtime_backward, # 7: n * len_seq (all same len)
                index_of_hidden_backward, # 8: n * len_seq (all same len)
                mask_obs, # 9: n * len_seq # 1.0 is observed events all others including BOS and EOS are all 0.0
                mask_unobs, # 10: n * len_seq # 1.0 is unobserved events all others including BOS and EOS are all 0.0
                log_censor_probs, # 11: n # log prob of THIS z is the missing part
                ]

    def calculate_log_censor_probs(self, obs_mask, events=None, time_stamps=None):
        """
        Calculate the log(p_censor(events, time_stamps)).
        :param torch.Tensor obs_mask: shape=[n_seq, len_seq].
        :param torch.Tensor events: shape=[n_seq, len_seq]. Optional.
        :param torch.Tensor time_stamps: shape=[n_seq, len_seq]. Optional.
        Note that for factorized missing mechanism, events is needed, but time_stamps isn't.
        Events could contains BOS, EOS, PAD. These tokens would be ignored
        during the calculation of p_censor.
        :rtype: torch.Tensor
        :return log_p_censor. shape=[n_seq]
        """
        return self.miss_mec.compute_probability(obs_mask, events, time_stamps)

    def augmentLogProbMissing(self, input):
        log_censor_probs = self.calculate_log_censor_probs(
            input[9], input[0], input[1] )
        input.append(log_censor_probs)
        return input

    def processBatchParticles(self, input):
        return self.funcBatch(input,
            idx_BOS=self.idx_BOS, idx_EOS=self.idx_EOS, idx_PAD=self.idx_PAD,
            device=self.device)

    #@profile
    def processBatchSeqsWithParticles(self, input):
        r"""
        batch of seqs, where each seq is many particles (as torch tensors)
        """
        batch_of_seqs = []
        for seq in input:
            batch_of_seqs.append(self.sampleForIntegral(
                seq, sampling=self.sampling, device=self.device) )
        return self.processBatchParticles(batch_of_seqs)


class DataProcessorNeuralHawkes(DataProcessorBase):
    def __init__(self, *args, **kwargs):
        super(DataProcessorNeuralHawkes, self).__init__(*args, **kwargs)

class DataProcessorNaive(DataProcessorBase):
    def __init__(self, *args, **kwargs):
        super(DataProcessorNaive, self).__init__('Naive', *args, **kwargs)


class LogWriter(object):

    def __init__(self, path, args):
        self.path = path
        self.args = args
        with open(self.path, 'w') as f:
            f.write("Training Log\n")
            f.write("Hyperparameters\n")
            for argname in self.args:
                f.write("{} : {}\n".format(argname, self.args[argname]))
            f.write("Checkpoints:\n")

    def checkpoint(self, to_write):
        with open(self.path, 'a') as f:
            f.write(to_write+'\n')

class LogReader(object):

    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            self.doc = f.read()

    def isfloat(self, str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    def casttype(self, str):
        res = None
        if str.isdigit():
            res = int(str)
        elif self.isfloat(str):
            res = float(str)
        elif str == 'True' or str == 'False':
            res = True if str == 'True' else False
        else:
            res = str
        return res

    def getArgs(self):
        block_args = self.doc.split('Hyperparameters\n')[-1]
        block_args = block_args.split('Checkpoints:\n')[0]
        lines_args = block_args.split('\n')
        res = {}
        for line in lines_args:
            items = line.split(' : ')
            res[items[0]] = self.casttype(items[-1])
        return res
