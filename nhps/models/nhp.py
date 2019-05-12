# -*- coding: utf-8 -*-
"""

neural Hawkes process (nhp) and continuous-time LSTM

@author: hongyuan
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from nhps.models.cont_time_cell import CTLSTMCell
from nhps.models.right2left_machine import Right2Left
from nhps.models.utils import makeHiddenForBound, makeHiddenForLambda

from torch.distributions.multinomial import Multinomial


class NeuralHawkes(nn.Module):

    def __init__(self, *, miss_mec=None,
        total_num, hidden_dim=32, beta=1.0, device=None):
        super(NeuralHawkes, self).__init__()

        self.total_num = total_num
        self.hidden_dim = hidden_dim

        self.idx_BOS = self.total_num
        self.idx_EOS = self.total_num + 1
        self.idx_PAD = self.total_num + 2

        self.beta = beta

        device = device or 'cpu'
        self.device = torch.device(device)
        self.right2left = False
        # this is ONLY turned ON when there is right to left machine
        # it is by default turned OFF

        self.mask_intensity = torch.ones(
            size=[self.total_num], dtype=torch.float32, device=self.device)

        self.Emb = nn.Embedding(
            self.total_num + 3, self.hidden_dim)
        # 0 -- K-1 : event types
        # K : BOS
        # K+1 : EOS
        # K+2 : PAD
        self.rnn_cell = CTLSTMCell(
            self.hidden_dim, beta=self.beta, device=self.device)
        self.hidden_lambda = nn.Linear(
            self.hidden_dim, self.total_num, bias=False)

        self.init_h = torch.zeros(size=[hidden_dim],
                                  dtype=torch.float32, device=self.device)
        self.init_c = torch.zeros(size=[hidden_dim],
                                  dtype=torch.float32, device=self.device)
        self.init_cb = torch.zeros(size=[hidden_dim],
                                   dtype=torch.float32, device=self.device)

        self.eps = np.finfo(float).eps
        self.max = np.finfo(float).max

        self.right2left_machine = None

        self.miss_mec = miss_mec

    def initBackwardMachine(self, *, type_back, hidden_dim_back, back_beta):
        self.right2left = True
        self.right2left_machine = Right2Left(hidden_dim_back, back_beta, self.total_num,
                                             self.device, type_back, self.hidden_dim)

    def setMaskIntensity(self, mask_intensity):
        #assert self.mask_intensity.device == neglect_mask.device, "Not same device?"
        assert self.mask_intensity.dtype == mask_intensity.dtype, "Not same dtype?"
        assert self.mask_intensity.size(0) == mask_intensity.size(0), "Not same size?"

        self.mask_intensity = mask_intensity

    def cuda(self, device=None):
        device = device or 'cuda:0'
        self.device = torch.device(device)
        assert self.device.type == 'cuda'
        super().cuda(self.device)

    def cpu(self):
        self.device = torch.device('cpu')
        super().cuda(self.device)

    def getStates(self, event, dtime):
        r"""
        go through the sequences and get all the states and gates
        we assume there is always a dim for particles but it can be 1
        Why so?
        Each complete seq x and z can be proposed result given x
        We can choose to propose only 1
        But when we propose 2, should the tensor have a new dimension
        just because one more particle is proposed?
        Definitely NOT!
        So we always have that dimension even though there's 1 particle
        """
        batch_size, num_particles, T_plus_2 = event.size()
        cell_t_i_minus = self.init_c.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.hidden_dim)
        cell_bar_im1 = self.init_cb.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.hidden_dim)
        hidden_t_i_minus = self.init_h.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.hidden_dim)

        all_cell, all_cell_bar = [], []
        all_gate_output, all_gate_decay = [], []
        all_hidden = []
        all_hidden_after_update = []

        for i in range(T_plus_2 - 1):
            # only BOS to last event update LSTM
            # <s> CT-LSTM

            emb_i = self.Emb(event[:, :, i ])
            dtime_i = dtime[:, :, i + 1 ]

            cell_i, cell_bar_i, gate_decay_i, gate_output_i = self.rnn_cell(
                emb_i, hidden_t_i_minus, cell_t_i_minus, cell_bar_im1
            )
            _, hidden_t_i_plus = self.rnn_cell.decay(
                cell_i, cell_bar_i, gate_decay_i, gate_output_i,
                torch.zeros_like(dtime_i)
            )
            cell_t_ip1_minus, hidden_t_ip1_minus = self.rnn_cell.decay(
                cell_i, cell_bar_i, gate_decay_i, gate_output_i,
                dtime_i
            )
            all_cell.append(cell_i)
            all_cell_bar.append(cell_bar_i)
            all_gate_decay.append(gate_decay_i)
            all_gate_output.append(gate_output_i)
            all_hidden.append(hidden_t_ip1_minus)
            all_hidden_after_update.append(hidden_t_i_plus)
            cell_t_i_minus = cell_t_ip1_minus
            cell_bar_im1 = cell_bar_i
            hidden_t_i_minus = hidden_t_ip1_minus
            # </s> CT-LSTM
        # these tensors shape : batch_size, num_particles, T+1, hidden_dim
        # cells and gates right after BOS, 1st event, ..., N-th event
        # hidden right before 1st event, ..., N-th event, End event (PAD)
        all_cell = torch.stack( all_cell, dim=2)
        all_cell_bar = torch.stack( all_cell_bar, dim=2)
        all_gate_decay = torch.stack( all_gate_decay, dim=2)
        all_gate_output = torch.stack( all_gate_output, dim=2)
        all_hidden = torch.stack( all_hidden, dim=2 )
        all_hidden_after_update = torch.stack( all_hidden_after_update, dim=2)

        return batch_size, num_particles, T_plus_2, \
        all_cell, all_cell_bar, all_gate_decay, all_gate_output, \
        all_hidden, all_hidden_after_update

    def getTarget(self, event, dtime):
        r"""
        make target variable and masks
        """
        batch_size, num_particles, T_plus_2 = event.size()
        mask_complete = torch.ones_like(dtime[:, :, 1:])

        target_data = event[:, :, 1:].detach().data.clone()

        mask_complete[target_data >= self.total_num] = 0.0

        target_data[target_data >= self.total_num] = 0 # PAD to be 0
        target = target_data
        return target, mask_complete

    def getSampledStates(
        self, dtime_sampling, index_of_hidden_sampling,
        all_cell, all_cell_bar, all_gate_output, all_gate_decay):
        r"""
        we output the sampled hidden states of the left-to-right machine
        states shape : batch_size * num_particles * T+1 * hidden_dim
        dtime_sampling : batch_size * num_particles * max_len_sampling
        index_of_hidden_sampling : batch_size * num_particles * max_len_sampling
        """
        batch_size, num_particles, T_plus_1, _ = all_cell.size()
        _, _, max_len_sampling = dtime_sampling.size()

        all_cell_sampling = all_cell.view(
            batch_size * num_particles * T_plus_1, self.hidden_dim )[
                index_of_hidden_sampling.view(-1), :].view(
                    batch_size, num_particles, max_len_sampling, self.hidden_dim)
        all_cell_bar_sampling = all_cell_bar.view(
            batch_size * num_particles * T_plus_1, self.hidden_dim )[
                index_of_hidden_sampling.view(-1), :].view(
                    batch_size, num_particles, max_len_sampling, self.hidden_dim)
        all_gate_output_sampling = all_gate_output.view(
            batch_size * num_particles * T_plus_1, self.hidden_dim )[
                index_of_hidden_sampling.view(-1), :].view(
                    batch_size, num_particles, max_len_sampling, self.hidden_dim)
        all_gate_decay_sampling = all_gate_decay.view(
            batch_size * num_particles * T_plus_1, self.hidden_dim )[
                index_of_hidden_sampling.view(-1), :].view(
                    batch_size, num_particles, max_len_sampling, self.hidden_dim)

        cy_sample, hy_sample = self.rnn_cell.decay(
            all_cell_sampling, all_cell_bar_sampling,
            all_gate_decay_sampling, all_gate_output_sampling,
            dtime_sampling
        )

        return hy_sample

    def getLambda(
        self, batch_size, num_particles, T_plus_2,
        target, mask_complete, mask_unobs, all_hidden, sampled_hidden ):
        r"""
        we output log_lambda for all cases:
        1. complete seq, i.e. union of partitions x and z
        2. unobs events, i.e. missing part z
        the 1. is used for computing p(x \sqcup z)
        the 2. is used to compute q(z | x) where q may not have right-to-left machine
        the lambda for 2. is called lambda_corrected
        because the belief states are corrected by the backward machine
        note that if all_hidden_back is not None
        the 2. should be computed using hidden states of both directions
        """

        all_lambda= F.softplus(self.hidden_lambda(all_hidden), beta=self.beta)
        log_lambda= torch.log(all_lambda+ self.eps)

        log_lambda_target = log_lambda.view(
            batch_size * num_particles * (T_plus_2 - 1), self.total_num
        )[
            torch.arange(0, batch_size * num_particles * (T_plus_2 - 1),
                         dtype=torch.int64, device=self.device),
            target.view( batch_size * num_particles * (T_plus_2 - 1) )
        ].view(batch_size, num_particles, T_plus_2 - 1)

        log_lambda_target_complete = log_lambda_target * mask_complete

        lambda_sum_complete = torch.sum(all_lambda, dim=3)
        log_lambda_sum_complete = torch.log(lambda_sum_complete + self.eps)
        log_lambda_sum_complete *= mask_complete

        log_lambda_target_unobs = log_lambda_target * mask_unobs
        lambda_sum_unobs = torch.sum(all_lambda, dim=3)
        log_lambda_sum_unobs = torch.log(lambda_sum_unobs + self.eps)
        log_lambda_sum_unobs *= mask_unobs

        all_lambda_sample = F.softplus(
            self.hidden_lambda(sampled_hidden), beta=self.beta )

        return log_lambda_target_complete, log_lambda_target_unobs, all_lambda_sample

    def getIntegral(
        self, lambda_sample, mask_sampling, duration ):
        r"""
        we output integral for both cases:
        1. p(x and z)---integral of total intensities conditioned on past
        2. q(z | x)---integral of total intensities conditioned on past and future
        """
        r"""
        mask_sampling : batch_size * num_particles * max_len_sampling
        duration : batch_size * num_particles
        """
        lambda_sample_sum = lambda_sample.sum(dim=3)
        lambda_sample_mean = torch.sum(
            lambda_sample_sum * mask_sampling, dim=2 ) / torch.sum(
            mask_sampling, dim=2 )
        integral = lambda_sample_mean * duration

        return integral

    def forward(self, input, mode=1, weight=None):

        event, dtime, post, duration, \
        dtime_sampling, index_of_hidden_sampling, mask_sampling, \
        event_obs, dtime_obs, \
        dtime_backward, index_of_hidden_backward, \
        dtime_backward_sampling, index_of_hidden_backward_sampling, \
        mask_obs_with_BOS, mask_unobs_with_BOS, logP_missingmec = input

        mask_obs = mask_obs_with_BOS[:, :, 1:]
        mask_unobs = mask_unobs_with_BOS[:, :, 1:]

        r"""
        event, dtime : batch_size, M, T+2
        post(erior of incomplete unobserved) : batch_size, M (not used)
        duration : batch_size, M
        dtime_sampling : batch_size, M, T_sample
        mode : see details in each mode
        Note: for log_likelihood, check if weight==None
        if False, do arithmetic average over particles
        """
        r"""
        note that dtime_backward and index_of_hidden_backward
        only have meaningful values for unobserved event tokens
        but dtime_backward_sampling and index_of_hidden_backward_sampling
        have meaningful values for both observed and unobserved event tokens
        however, all of these are only used when computing q(z | x)
        """

        batch_size, num_particles, T_plus_2, \
        all_cell, all_cell_bar, all_gate_decay, all_gate_output, \
        all_hidden, all_hidden_after_update = self.getStates(event, dtime)

        target, mask_complete = self.getTarget( event, dtime )

        sampled_hidden = self.getSampledStates(
            dtime_sampling, index_of_hidden_sampling,
            all_cell, all_cell_bar, all_gate_output, all_gate_decay
        )

        # <s> \lambda_{k_i}(t_i | H_i) for events
        log_lambda_target_complete, log_lambda_target_unobs, \
        all_lambda_sample = self.getLambda(
            batch_size, num_particles, T_plus_2,
            target, mask_complete, mask_unobs, all_hidden, sampled_hidden )
        # batch_size * num_particles * T_plus_2-1
        # </s> \lambda_{k_i}(t_i) for events

        # <s> utils for right2left machine
        if self.right2left:

            r"""
            right-to-left machine is working, which means it is
            1. used while q(z | x) is being trained
            OR
            2. used while q(z | x) is proposing by particle smoothing
            """

            T_obs_plus_2, \
            all_cell_obs, all_cell_bar_obs, all_gate_decay_obs, all_gate_output_obs, \
            all_hidden_obs, all_hidden_after_update_obs = self.right2left_machine(
                event_obs, dtime_obs )

            log_lambda_corrected_target_unobs, \
            all_lambda_corrected_sample = self.right2left_machine.getLambdaCorrected(
                all_hidden, sampled_hidden,
                dtime_backward, index_of_hidden_backward,
                dtime_backward_sampling, index_of_hidden_backward_sampling,
                target, mask_unobs,
                all_cell_obs, all_cell_bar_obs,
                all_gate_output_obs, all_gate_decay_obs,
                self.hidden_lambda
            )

        else:

            r"""
            right-to-left is not working, so
            1. q(z | x) is only a particle filter
            2. q(z | x) is a stripped version of neural hawkes p(x and z)
            3. q(z | x) has NO extra params than p(x and z)
            """
            log_lambda_corrected_target_unobs = log_lambda_target_unobs
            all_lambda_corrected_sample = all_lambda_sample
        # </s>

        # <s> int_{0}^{T} lambda_sum dt for events
        integral_of_lambda_complete = self.getIntegral(
            all_lambda_sample, mask_sampling, duration )

        integral_of_lambda_correctd_unobs = self.getIntegral(
            all_lambda_corrected_sample, mask_sampling, duration )

        # batch_size * num_particles
        # </s> int_{0}^{T} lambda_sum dt for events

        # <s> log likelihood computation

        logP_complete = log_lambda_target_complete.sum(2) - integral_of_lambda_complete
        # log p( x and z ) --- according to (trained) neural Hawkes process

        logP_unobs = log_lambda_corrected_target_unobs.sum(2) - integral_of_lambda_correctd_unobs
        # log q( z | x ) --- proposal distribution
        # batch_size * num_particles
        # </s> log likelihood computation

        r"""
        for any seq in a batch, there are some particles (can be 1)
        there are (normalized) weight for each particle
        if weight is not given, i.e., None
        we assume uniform weight, i.e., average over particles
        """

        if weight is None:
            weight = torch.ones(size=[batch_size, num_particles],
                               dtype=torch.float32, device=self.device)
            weight = weight / torch.sum(weight, dim=1, keepdim=True)

        if mode == 1:
            # complete log likelihood
            objective = -torch.sum( logP_complete * weight )
            num_events = torch.sum( mask_complete )

        elif mode == 2:
            # incomplete log likelihood for observed events
            raise Exception("This mode is useless.")

        elif mode == 3:
            # incomplete log likelihood for unobserved events
            # without considering missing mechanism
            # used to compute how probable the given unobserved is under model
            # whether have F_j depends on self.right2left
            # if self.right2left == True, F_j is used
            r"""
            this mode will not likely be used
            """
            objective = -torch.sum( logP_unobs * weight )
            num_events = torch.sum( mask_unobs )

        elif mode == 4:
            r"""
            according to the general framework
            to compute the weights
            we should consider the missingness mechanism
            which is part of input in the log space
            for details, refer to the paper
            """
            #normalized weights for each particle
            logP_diff = logP_missingmec + logP_complete - logP_unobs

            assert logP_diff.dim()==2, "logP_diff should be a matrix!"
            assert logP_diff.size(0)==batch_size, "dim-0 batch size"
            assert logP_diff.size(1)==num_particles, "dim-1 num particles"
            objective = F.softmax( logP_diff, dim=1 )
            r"""
            this step is computing weights and doing normalization
            for MBR, normalization is not necessary in theory
            but in practice, normalization can help with computation
            why?
            because logP_diff can be either very large and very small
            if we do not normalize it
            taking exp may give us overflow or underflow
            so all particles may have \inf or 0.0 for weights
            but if we normalize it, it is equal to doing softmax
            and after (a well-implemented) softmax
            all that matters is the relative difference
            (with max diff 1.0-0.0=1.0)
            so we should normalize it
            we can also have this argument in paper
            """
            num_events = None

        elif mode == 5:

            raise Exception("MCEM is not used for this paper")
            # compute ELBO of log p(x) assuming MAR
            # see paper for more details
            # Note that we need to optimize this ELBO because the original obj
            # easily gets stuch with bad local minima that only maximizes MAX
            # this mode is used for training

            logP_diff = logP_complete - logP_unobs.detach()
            assert logP_diff.dim()==2, "logP_diff should be a matrix!"
            assert logP_diff.size(0)==batch_size, "dim-0 batch size"
            assert logP_diff.size(1)==num_particles, "dim-1 num particles"
            objective = -torch.sum( logP_diff.mean( dim=1 ) )
            num_events = torch.sum( mask_obs )

        elif mode == 6:

            raise Exception("MCEM is not used for this paper")
            # compute estiamted p(x) assuming MAR
            # log p(x) = log sum exp (beta - maxbeta) + maxbeta - log M
            # beta = log p(s,u) - log q(u | s)
            #
            # BUT Note that this log sum exp could easily be dominated by
            # MAX particle so only the MAX one gets optimized
            # and the mean can go anywhere --- large variance in nature
            # so this mode is ONLY used for eval

            logP_diff = logP_complete - logP_unobs.detach()
            assert logP_diff.dim()==2, "logP_diff should be a matrix!"
            assert logP_diff.size(0)==batch_size, "dim-0 batch size"
            assert logP_diff.size(1)==num_particles, "dim-1 num particles"
            logP_diff_max, _ = torch.max( logP_diff, dim=1, keepdim=True )
            vec_num_particles = torch.empty(size=[batch_size], dtype=torch.float32,
                                            device=self.device).fill_(num_particles*1.0)
            est_logPx = torch.log(
                torch.sum(
                    torch.exp(
                        logP_diff - logP_diff_max
                ), dim=1 ) ) + logP_diff_max.squeeze() - torch.log(vec_num_particles)
            objective = -torch.sum( est_logPx )
            num_events = torch.sum( mask_obs )

        elif mode == 7:
            r"""
            Inclusive KL divergence
            KL(Pr | q)
            q(z | x) --- the proposal distribution
            Pr(z | x, x is the observed part) --- the true probability dist
            Precise derivation can be found in paper
            """
            inclusiveKL = torch.mean( -logP_unobs, dim=1 )
            objective = torch.mean( inclusiveKL )
            r"""
            revisit how to get KL div per token
            """
            num_events = torch.sum( mask_unobs )
            r"""
            why count the num of unobserved events?
            because it is KL divergence of q(u | s) and Pr(u | s)
            so to get per token distance, we should divide by num of unobs
            """

        elif mode == 8:
            r"""
            Exclusive KL divergence
            KL(q | Pr)
            """
            distance = logP_unobs - logP_complete.detach() - logP_missingmec
            exclusiveKL = torch.mean( 0.5 * distance * distance, dim=1 )
            r"""
            since this is noisy, we can add a baseline to it
            we can use the same exp baseline as Lin and Eisner NAACL 2018 did
            But we can also consider
            Pr(x, x is the observed part)
            as a natural baseline
            To compute this baseline also needs all the particles and
            their prob under known missingness mechanism
            """
            objective = torch.mean( exclusiveKL )
            r"""
            revisit how to get KL div per token
            """
            num_events = torch.sum( mask_unobs )

        elif mode == 9:
            r"""
            log q(z | x) -- the proposal distribution
            this mode is special:
            it does not output a scalar,
            it outputs a vector of batch_size
            """
            assert logP_unobs.size(1) == 1, "Only one partition when this mode is used"
            assert mask_unobs.size(1) == 1, "Only one partition when this mode is used"
            objective = torch.mean( logP_unobs, dim=1 )
            num_events = mask_unobs.sum(dim=2).sum(dim=1)

        else:
            raise Exception( "Unknown mode : {}".format(mode) )

        return objective, num_events

    def resetPolicy(self, particle_num):
        self.states = self.rnn_cell(
            self.Emb(
                torch.empty(size=[particle_num], dtype=torch.int64,
                            device=self.device).fill_(self.idx_BOS)
            ),
            self.init_h.unsqueeze(0).expand(particle_num, self.hidden_dim),
            self.init_c.unsqueeze(0).expand(particle_num, self.hidden_dim),
            self.init_cb.unsqueeze(0).expand(particle_num, self.hidden_dim)
        )

    def update(self, input):
        r"""
        a batch of multiple particles
        """
        types_to_update, dtimes_to_update, updated = input
        # particle_num

        cell_t_ip1_minus, hidden_t_ip1_minus = self.rnn_cell.decay(
            self.states[0], self.states[1],
            self.states[2], self.states[3],
            dtimes_to_update
        )

        states_candidate = self.rnn_cell(
            self.Emb(types_to_update),
            hidden_t_ip1_minus, cell_t_ip1_minus, self.states[1]
        )

        self.states[0][updated.data, :] = states_candidate[0][updated.data, :]
        self.states[1][updated.data, :] = states_candidate[1][updated.data, :]
        self.states[2][updated.data, :] = states_candidate[2][updated.data, :]
        self.states[3][updated.data, :] = states_candidate[3][updated.data, :]

        return hidden_t_ip1_minus

    def computeLambda(self, compute_for_targetdist, dtimes, dtimes_back):
        """
        this function is primarily used in proposing
        but if we use resampling, we need weight along with proposing
        so we also need to know lambda p
        this is decided by compute_for_targetdist
        if it is True, then we compute lambda p and no dtimes_back is used
        otherwise, we compute lambda q
        and then when self.right2left == True, it is smoothing, dtimes_back is used
        """
        if compute_for_targetdist == True:
            # compute for target distribution p
            all_lambda = self.computeLambdaTarget(
                self.hidden_lambda, self.states, dtimes)
        elif compute_for_targetdist == False:
            # compute lambda q
            all_lambda = self.computeLambdaProposal(
                self.hidden_lambda, self.states, dtimes, dtimes_back)
        else:
            raise TypeError('compute_for_targetdist has to be boolean')

        return all_lambda.detach().data

    def computeTotalLambda(self, compute_for_targetdist, dtimes, dtimes_back=None):

        all_lambda = self.computeLambda(
            compute_for_targetdist, dtimes, dtimes_back)
        all_total_lambda = all_lambda.sum(dim=2)
        return all_total_lambda.detach().data

    def getBound(
        self, hidden_lambda, states, dtime, particle_num,
        dtime_back=None):
        r"""
        work with a batch of particles
        output torch tensor data
        """
        # batch version of computing lambda bound
        linear_weight = hidden_lambda.weight.transpose(0, 1)
        # shape: (2*)hidden_dim * total_num
        hy_forward = makeHiddenForBound(
            linear_weight, states, dtime, particle_num, self.hidden_dim, self.total_num)

        linear_weight = linear_weight.unsqueeze(dim=0).expand(
            particle_num, *linear_weight.size())

        projected_hidden = (hy_forward * linear_weight).sum(dim=1)

        if self.right2left:
            assert dtime_back is not None, "No time for backward?"
            lambda_bound = self.right2left_machine.pack_bound(
                hidden_lambda, states, dtime, particle_num,
                dtime_back, projected_hidden
            )
        else:
            lambda_bound = F.softplus(projected_hidden, beta=self.beta)

        return lambda_bound.detach()

    def computeLambdaTarget(self, hidden_lambda, states, dtime):
        """
        only used to compute lambda of target distribution
        i.e. lambda p
        """
        hy_forward = makeHiddenForLambda(states, dtime)
        # particle_num * num * hidden_dim
        projected_hidden = hidden_lambda(hy_forward)
        lambda_t = F.softplus(projected_hidden, beta=self.beta )
        return lambda_t.detach().data

    def computeLambdaProposal(
        self, hidden_lambda, states, dtime
            , dtime_back=None):

        """
        only used to propose
        only compute lambda of proposal distribution
        regardless of filtering or smoothing
        """

        hy_forward = makeHiddenForLambda(states, dtime)
        # particle_num * num * hidden_dim
        projected_hidden = hidden_lambda(hy_forward)

        if self.right2left:
            assert dtime_back is not None, "No time for backward?"

            lambda_t = self.right2left_machine.pack_lambda(
                hidden_lambda,
                dtime_back, hy_forward, projected_hidden
            )

        else:

            lambda_t = F.softplus(projected_hidden, beta=self.beta )
        # particle_num * num * total_num
        return lambda_t.detach().data

    def sample_unobs_batch(self, dtime_bound, particle_num,
        num=100, verbose=False, resamplng=False):
        r"""
        1. we do sampling in a batch
        2. we sample for diff particles independently
        2.1. compute their U, E, and lambda_t (vecs) in parallel (thanks to GPU)
        2.2. reject and accept for each particle
        2.3. find earliest accepted proposals
        2.4. if none is accepted for any particle, recompute again
        2.5. dtime > threshold is not considered here, it is in sample_particles_batch
        """
        """
        RESAMPLING
        Note this method only draws an accepted event for each particle
        that may sometimes reach out of the current interval
        """
        if verbose:
            print("Sample unobserved events with thinning algo in a batch")

        if self.right2left:

            lambda_bound = self.getBound(
                self.hidden_lambda,
                self.states,
                torch.zeros(size=[particle_num], dtype=torch.float32, device=self.device),
                particle_num,
                torch.zeros(size=[particle_num], dtype=torch.float32, device=self.device)
            )

        else:

            lambda_bound = self.getBound(
                self.hidden_lambda,
                self.states,
                torch.zeros(size=[particle_num], dtype=torch.float32, device=self.device),
                particle_num
            )

        lambda_bound *= self.mask_intensity.unsqueeze(dim=0)

        lambda_bound_sum = torch.sum(
            lambda_bound, dim=1, keepdim=True)#.detach()

        accepted_dtimes = torch.zeros(size=[particle_num],
                                      dtype=torch.float32, device=self.device)
        #accepted_ids = self.device.LongTensor(self.particle_num)
        accepted_types = torch.empty(size=[particle_num], dtype=torch.int64,
                                     device=self.device).fill_(self.idx_PAD)

        accepted_log_lambda_p = torch.zeros(
            size=[particle_num], dtype=torch.float32, device=self.device)
        accepted_log_lambda_q = torch.zeros(
            size=[particle_num], dtype=torch.float32, device=self.device)

        accepted_logPs = torch.zeros(size=[particle_num], dtype=torch.float32,
                                     device=self.device)

        finished = torch.zeros(size=[particle_num], dtype=torch.uint8,
                               device=self.device)
        E = torch.empty(size=[particle_num, num], dtype=torch.float32, device=self.device)
        U = torch.empty(size=[particle_num, num], dtype=torch.float32, device=self.device)

        ranger = torch.arange(0, particle_num, dtype=torch.int64, device=self.device)

        cnt_step = 0

        r"""
        this while loop is sometimes hard to stop
        because intensity (lambda) very small, compared to lambda_bound
        we need to do things to let it finish within reasonable time
        1. we adopt precedural recipe (as discussed in NIPS 17)
        1.1. we sample Delta time ~ Exp(lambda_bound_sum) and cumsum them
        1.2. if not accepted, we add largest dtime to accepted_dtimes
        2. we use a dtime bound for this sampling
        2.1. if each particle reaches the dtime bound before accepted, it stops
        because we only care unobserved events in intervals !
        """

        while not finished.all():
            """
            finish when all particles are finished with
            drawing a new accepted event
            """
            if verbose:
                #print("{}-th step in sample_unobs_batch".format(cnt_step))
                cnt_step += 1

            E.exponential_(1.0)
            U.uniform_(0.0, 1.0)
            mat_dtime = E / lambda_bound_sum
            # particle_num * num(=100)
            mat_dtime = mat_dtime.cumsum(dim=1)#.detach()

            # mat_dtime added by current accum dtime
            mat_dtime += accepted_dtimes.unsqueeze(1)#.detach()

            """
            although we only need to know lambda q to propose events
            we still need to compute weight along the way
            so we need both lambda p and lambda q
            we also need missing factor
            but this factor is handled out of this function
            """
            # lambda for target dist
            mat_lambda_t_p = self.computeLambdaTarget(
                self.hidden_lambda, self.states, mat_dtime)
            # lambda for proposal dist
            if self.right2left:

                mat_dtime_back = dtime_bound - mat_dtime
                mat_dtime_back[ mat_dtime_back < 0.0 ] = 0.0

                mat_lambda_t = self.computeLambdaProposal(
                    self.hidden_lambda, self.states, mat_dtime,
                    mat_dtime_back )

            else:

                mat_lambda_t = self.computeLambdaProposal(
                    self.hidden_lambda, self.states, mat_dtime )

            # lambda of target dist do not need this mask
            mat_lambda_t *= self.mask_intensity.unsqueeze(dim=0).unsqueeze(dim=0)

            # particle_num * num(=100) * self.unobs_num
            u = U * lambda_bound_sum / torch.sum(mat_lambda_t, dim=2)
            # particle_num * num(=100)
            min_u_each_particle, _ = u.min(dim=1)
            indices_accepted_particles = min_u_each_particle < 1.0
            not_accepted = ~indices_accepted_particles
            # we find accepted particles
            # but we only want to update these are not updated yet
            # i.e. we need to know which are already finished
            not_finished = ~finished
            indices_tobe_updated = indices_accepted_particles & not_finished

            # before we update those to be updated
            # we first find non-accepted and non-finished
            # and accumulate dtime on those particles
            # -1 because that is the index of finally accumulated dtimes
            accepted_dtimes[not_finished & not_accepted] = \
            mat_dtime[:, -1][not_finished & not_accepted]#.detach()

            # for the accepted proposals ( i.e. u < 1.0 )
            # find their accepted dtimes and lambdas
            mat_dtime_with_unacc_large = mat_dtime.clone()
            mat_dtime_with_unacc_large[ u >= 1.0 ] = mat_dtime.max() + 1.0
            # particle_num * num
            dtime_each_particle, dtime_id_each_particle = \
            mat_dtime_with_unacc_large.min(dim=1)
            # particle_num
            accepted_dtimes[indices_tobe_updated] = \
            dtime_each_particle[indices_tobe_updated]#.detach()

            lambda_t_p_each_particle = mat_lambda_t_p[
                ranger, dtime_id_each_particle, :]
            lambda_t_each_particle = mat_lambda_t[
                ranger, dtime_id_each_particle, :]#.detach()
            # particle_num * self.unobs_num

            max_lambda_each_particle, indices_max_lambda_each_particle = \
            lambda_t_each_particle.max(dim=1)

            """
            according to which event type is generated,
            we need to find their lambda p
            """
            chosen_lambda_p_each_particle = lambda_t_p_each_particle[
                ranger, indices_max_lambda_each_particle]

            accepted_types[indices_tobe_updated] = \
            indices_max_lambda_each_particle[indices_tobe_updated]#.detach()


            """
            compute log_lambda_p and log_lambda_q
            """
            log_lambda_p = torch.log(chosen_lambda_p_each_particle + self.eps)
            accepted_log_lambda_p[indices_tobe_updated] =\
            log_lambda_p[indices_tobe_updated]
            log_lambda_q = torch.log(max_lambda_each_particle + self.eps)
            accepted_log_lambda_q[indices_tobe_updated] =\
            log_lambda_q[indices_tobe_updated]

            # once they are updated, then they are marked as finished
            finished[indices_accepted_particles] = 1

            # even they are not updated, if they go out of boundary
            # they are marked as finished
            finished[accepted_dtimes >= dtime_bound] = 1
            # in this case, we do not need to change types or weight-related because
            # these out-of-bound particles NOT used out of this function anyway

        return accepted_types, accepted_dtimes, accepted_log_lambda_p, accepted_log_lambda_q

    def initResampling(self, one_seq, num_points_per_obs_for_integral,
        minimum_num_points_per_interval):
        """
        sample points for each interval
        we choose a total # of points to use for the entire duration
        then we set # of points for each interval prop to its length
        we can propose max(somenumber, dt/T * N) time points each interval
        note: there should be at least one point in each interval
        """
        total_points = num_points_per_obs_for_integral * (len(one_seq) - 1)
        num_points_for_integral = [1] * (len(one_seq)-1)
        total_t = one_seq[-1]['time_since_start'] - one_seq[0]['time_since_start']
        total_t = max(1e-5, total_t)
        for idx_interval in range(len(one_seq) - 1):
            time_ratio = (one_seq[idx_interval+1]['time_since_last_event']) / total_t
            num_points_this_interval = max(minimum_num_points_per_interval, int(time_ratio * total_points))
            num_points_for_integral[idx_interval] = num_points_this_interval
        return num_points_for_integral

    def prepareResamplingInterval(self,
        num_points_this_interval, item, particle_num):
        """
        sample time points for this interval
        """
        sampled_points_integral = torch.rand([num_points_this_interval], device=self.device)
        sampled_points_integral, _ = sampled_points_integral.sort()
        sampled_points_integral *= float(item['time_since_last_event'])
        """
        init with the number of points sampled for the integral.
        they're all zeros when finished.
        """
        points_not_covered = torch.full(
            size=[particle_num], device=self.device,
            fill_value=num_points_this_interval, dtype=torch.int64)
        """
        index_not_covered used to track
        which sampled points are not covered for each particle
        1 : not covered yet
        0 : already covered
        """
        index_not_covered = torch.ones(
            size=[particle_num, num_points_this_interval],
            device=self.device, dtype=torch.uint8)
        """
        init weight for this segment
        the log weight should be
        log pmiss + log p - log q
        as shown in our algorithm 1 in paper
        log pmiss is init as all 0
        but will be filled after this interval proposing finished
        2. for log p(x, z)
        we starts from the i-th observed event (even if it is BOS)
        and adds up their log lambda p
        also, we minus their integral term in the end
        3. for log q(z | x)
        we starts from the 1st proposed event (there may be no proposed events)
        and adds up their log lambda q
        also, we minus their integral term in the end
        """
        """
        for total lambda of p and q
        """
        total_lambda_p_atsampledpoints = torch.zeros(
            size=[particle_num, num_points_this_interval],
            dtype=torch.float32, device=self.device)
        total_lambda_q_atsampledpoints = torch.zeros(
            size=[particle_num, num_points_this_interval],
            dtype=torch.float32, device=self.device)

        return points_not_covered, index_not_covered, \
        total_lambda_p_atsampledpoints, total_lambda_q_atsampledpoints, \
        sampled_points_integral

    def printDebug(self, cnt_call, i_item, total_dtime, dtime_bound,
        cnt, max_len, types, dtimes, logPs):
        print("\n\n")
        print(
            "{}-th call of sample_unobs_batch for {}-th item".format(
                cnt_call, i_item))
        print("\n")
        print("current total_dtime is {}".format(total_dtime.cpu().numpy()))
        print("dtime bound is {}".format(dtime_bound))
        print("current cnt is {}".format(cnt.cpu().numpy()))
        print("max_len is {}".format(max_len))
        print("\n")
        print("after this call, the types, dtimes and logPs are:")
        print("types {}".format(types.cpu().numpy()))
        print("dtimes {}".format(dtimes.cpu().numpy()))
        print("logPs {}".format(logPs.cpu().numpy()))
        print("\n")

    def computeLambdaAtSampledTimes(self,
        sampled_points_integral, num_points_this_interval,
        particle_num, dtimes, dtime_bound, total_dtime, finished,
        total_lambda_p_atsampledpoints, total_lambda_q_atsampledpoints,
        points_not_covered, index_not_covered
        ):
        # track the diff :
        # sampled time points minus current total dtime
        # note: newly generated&accepted dtime not used yet
        integral_dtime = sampled_points_integral - total_dtime.repeat(num_points_this_interval)\
            .reshape(num_points_this_interval, particle_num).transpose(0, 1)
        # if it is > 0
        # sampled time points larger
        # meaning they are not covered by prev dtimes yet
        left_integral_mask = integral_dtime > 0
        # if it is < newly generated/accepted dtime
        # sampled time points are covered by this new dtime
        # so we have to compute the lambdas at them now
        # otherwise, they will not be toched again when the algo proceeds
        right_integral_mask = integral_dtime <\
                              dtimes.unsqueeze(1).expand(particle_num, num_points_this_interval)
        # shape = [particle_num, num_points_this_interval
        # if both true, then we will compute lambdas
        # for these time points
        # moreover, we only compute for particles
        # that have not reached max len
        # because for those, proposed dtime may be meaningless
        # we will handle the particles that have reached max len
        # outside of while loop
        both_left_and_right = left_integral_mask & right_integral_mask
        not_finished = (~finished).unsqueeze(1).expand(
            particle_num, num_points_this_interval)
        integral_mask = both_left_and_right & not_finished
        # The following line might be able to replace the not_finished and left_mask.
        # integral_mask = both_left_and_right & index_not_covered & integral_mask
        # compute total intensity at these times
        # using the mask
        # we do not care whatever values masked out though
        """
        NOTE: consider dtime_back if self.right2left is True
        we only care wherever mask is 1
        we do not care where mask is 0
        This is important for particle smoothing!!!!!!
        also note that: unlike in sampled_unobs_batch function
        here we use sampled time points instead of proposed events
        so all dtimes_back should be > 0
        """
        sampled_points_back = dtime_bound - sampled_points_integral
        sampled_points_back = sampled_points_back.repeat(particle_num).reshape(particle_num, -1)
        assert (sampled_points_back < 0.0).sum() < 1, "sampled points back not > 0?"

        if torch.any(integral_mask):
            total_lambda_p_atsampledpoints[integral_mask] =\
            self.computeTotalLambda(
                True, integral_dtime)[integral_mask]
            total_lambda_q_atsampledpoints[integral_mask] =\
            self.computeTotalLambda(
                False, integral_dtime, sampled_points_back)[integral_mask]
        """
        for each dtime, it may jump over 0, 1, 2, ... time points
        each particle may jump over diff # of points
        so we may need
        a residual machine that tracks how many points left for each particle
        we call it points_not_covered
        it tracks # sampled points not covered by current total dtime
        for each proposed event, we need to update this tensor
        we should just count # of sampled points be covered by this proposed event
        sum them, and then substract from points_not_covered
        it can also be used as sanity check :
        sometimes, when finished, all sampled points are covered
        sometimes, when finished, some particle reached max num
        we need to deal with these 2 cases
        not very tricky though
        """
        points_not_covered -= torch.sum(integral_mask, dim=1)
        index_not_covered[integral_mask] = 0

    def updateSumLogLambda(self,
        sum_log_lambda_p, sum_log_lambda_q,
        log_lambda_p, log_lambda_q,
        indices_tobe_updated):
        """
        for this proposed event generated by thinning algorithm
        we need to update the sum_log_lambda_p and _q
        we only add up the log lambda
        if the dtime of this particle does not exceed this interval
        and it does not exceed max len
        """
        sum_log_lambda_p[indices_tobe_updated] += \
        log_lambda_p[indices_tobe_updated]
        sum_log_lambda_q[indices_tobe_updated] += \
        log_lambda_q[indices_tobe_updated]
        # Nothing should necessarily be returned -- all the operations are in-place.

    def computeIntegralInterval(self,
        points_not_covered, index_not_covered,
        sampled_points_integral, total_dtime, dtime_bound,
        num_points_this_interval, particle_num,
        total_lambda_p_atsampledpoints,
        total_lambda_q_atsampledpoints):
        """
        integral computation for this interval is supposed to finished now
        but because some particles may reach max len
        so there might be some sampled points
        that are not covered yet
        we need to find them and then
        update their total_lambda_p and _q
        """
        assert points_not_covered.sum() >= 0, "over covered?"
        integral_dtime = sampled_points_integral - \
        total_dtime.repeat(num_points_this_interval).reshape(
            num_points_this_interval, particle_num).transpose(0, 1)
        # assert (integral_dtime[index_not_covered] >= 0.0).all(), "non-covered sampled points are covered? WTH..."
        sampled_points_back = dtime_bound - sampled_points_integral
        sampled_points_back = sampled_points_back.repeat(particle_num).reshape(particle_num, -1)
        total_lambda_p_atsampledpoints[index_not_covered] = \
        self.computeTotalLambda(
            True, integral_dtime)[index_not_covered]
        # note that we can input time_back anyway
        # because if it is not smoothing , i.e. filtering
        # then right to left machine is not used
        # so time_back is not used anyway
        total_lambda_q_atsampledpoints[index_not_covered] = \
        self.computeTotalLambda(
            False, integral_dtime, sampled_points_back)[index_not_covered]
        points_not_covered -= torch.sum(index_not_covered, dim=1)
        assert (points_not_covered == 0).all(), "not all sampled time points be covred?"
        """
        so far, all sampled time points are covered
        so we can compute integrals for both p and q for each particle
        """
        integral_p = total_lambda_p_atsampledpoints.mean(1) * dtime_bound
        integral_q = total_lambda_q_atsampledpoints.mean(1) * dtime_bound
        return integral_p, integral_q

    def computeWeight(self, i_item, len_seq,
        hy_forward_befre_obs, type_event_obs,
        sum_log_lambda_p, sum_log_lambda_q, integral_p, integral_q,
        log_pmiss, log_weights
        ):
        """
        if this observed event is not EOS
        we should consider it in log_p
        why so?
        because target distribution is over complete data
        then why not we use the left boundary (i.e. starting obs event)
        for this interval
        why the end?
        because by resampling, we want to keep the particles
        with the proposed events that can well explain the FUTURE
        so we should take some future into account of resampling
        otherwise, we resample based on only past
        and this resampling may even hurt results
        see algo-1 in the paper for more details
        """
        if i_item < len_seq - 2:
            """
            compute lambda p for this obs
            """
            projected_hy = self.hidden_lambda(hy_forward_befre_obs)
            all_lambda_t_p = F.softplus(projected_hy, beta=self.beta )
            lambda_t_p_each_particle = all_lambda_t_p[:, type_event_obs]
            log_lambda_p = torch.log(
                lambda_t_p_each_particle + self.eps)
            sum_log_lambda_p += log_lambda_p
        """
        now we already finished sum_log_lambda_p
        compute log_weight = log_pmiss + log_p - log_q
        """
        log_p = sum_log_lambda_p - integral_p
        log_q = sum_log_lambda_q - integral_q
        log_weights[:] = log_pmiss + log_p - log_q
        return sum_log_lambda_p

    def doMultinomialResampling(self, log_weights, resampling_variables):
        # The following design is for efficiency.
        # There're three kinds of particles:
        # (1) Some particles generate exactly 1 particle.
        # (2) Some particles might generate more than 1 particles. (2, 3, 4, ...)
        # (3) Some particles are eliminated, i.e., generate 0 particle.
        # Our design is, the particles generating exactly 1 particle should remain unchanged,
        # and the particles generating more than 1 particles should "send" particles
        # to the positions that vacant. (These positions are vacant since their original
        # occupiers were eliminated.)
        # In order to avoid for loop, I implement all of above with matrix operations.
        # E.g., suppose we have a particle set containing 6 particles, and they generate
        # 0, 3, 1, 0, 2, 0 particle(s) individually (note that 3 + 1 + 2 == 6)
        # In my implementation, the particles indexed [1] and [4] should send 3 particles (2 from [1]
        # and 1 from [4]) to positions [0], [3], [5]
        # This is, probably, the most efficient way.

        particle_num = log_weights.shape[0]

        weights = torch.exp(log_weights).detach()
        weights = weights / weights.sum()
        weights_cum = weights.cumsum(dim=0)
        weights_cum[-1] = 1 + 1e-5
        choices = weights_cum.repeat(particle_num).reshape(particle_num, particle_num).cpu().numpy()
        random_numbers = np.random.rand(particle_num)
        choices = (choices.T < random_numbers).sum(axis=0)
        choices_cnt = np.zeros(shape=[particle_num, particle_num], dtype=np.int64)
        choices_cnt[np.arange(particle_num), choices] = 1
        choices_cnt = choices_cnt.sum(axis=0)

        arange_mat = np.arange(particle_num).repeat(particle_num).reshape(particle_num, particle_num).T

        positions_available = choices_cnt - 1
        positions_available[positions_available < 0] = 0

        positions_available_ext = positions_available.repeat(particle_num)\
            .reshape(particle_num, particle_num)
        sender_indices = list(arange_mat.T[arange_mat < positions_available_ext])
        receiver_indices = list(np.arange(particle_num)[choices_cnt == 0])

        assert len(sender_indices) == len(receiver_indices) == positions_available.sum(), 'Sanity check'

        for var_ in resampling_variables:
            assert isinstance(var_, torch.Tensor), "Give me tensor please."
            assert var_.shape[0] == particle_num, "The 1st dimension of tensor must be particle_num"
            var_[receiver_indices] = var_[sender_indices]

    @staticmethod
    def need_resampling_now(log_weights, threshold_ratio, particle_num):
        """
        Judge whether do resampling or not.
        :param torch.Tensor log_weights:
        :param float threshold_ratio:
        :param int particle_num:
        :rtype: bool
        """
        weights = torch.exp(log_weights)
        effective_sampling_size = (weights.sum()**2 / (weights**2).sum())
        threshold = threshold_ratio * particle_num
        return effective_sampling_size < threshold

    def sample_particles(self,
        particle_num, one_seq, num_unobs_tokens,
        num_points_per_obs_for_integral=1,
        num=100, verbose=False,
        resampling=False,
        minimum_num_points_per_interval=1,
        resampling_threshold_ratio=0.5,
        need_eliminate_log_base=False):
        """

        :param int particle_num:
        :param list one_seq:
        :param int num_unobs_tokens:
        :param int num:
        :param bool verbose:
        :param bool resampling: It is default True.
        :param int num_points_per_obs_for_integral: It is set as 1 by default.
        :param int minimum_num_points_per_interval: It is set as 1 by default.
        :param float resampling_threshold_ratio: I use w.sum()**2 / (weights**2).sum() to calculate
        the effective sampling size. If it is below resampling_threshold_ratio * particle_num, resampling will be
        applied. Else, the log probabilities will be accumulated.
        :return:
        """

        if resampling:
            assert self.miss_mec is not None, "you must provide miss-mec for resampling"
            num_points_for_integral = self.initResampling(
                one_seq, num_points_per_obs_for_integral,
                minimum_num_points_per_interval)

        len_seq = len(one_seq)

        max_len = num_unobs_tokens

        duration = torch.empty(size=[particle_num], dtype=torch.float32,
                               device=self.device).fill_(float(one_seq[-1]['time_since_start']))

        events_particles = torch.empty(size=[particle_num, len_seq+max_len],
                                       dtype=torch.int64, device=self.device).fill_(self.idx_PAD)
        events_particles[:, 0].fill_(self.idx_BOS)
        # events particles do not have EOS
        dtimes_particles = torch.zeros(size=[particle_num, len_seq+max_len],
                                       dtype=torch.float32, device=self.device)

        events_obs = torch.empty(size=[particle_num, len_seq], dtype=torch.int64, device=self.device)
        events_obs.fill_(self.idx_PAD)
        events_obs[:, 0].fill_(self.idx_BOS)
        events_obs[:, -1].fill_(self.idx_EOS)
        dtimes_obs = torch.zeros(size=[particle_num, len_seq], dtype=torch.float32, device=self.device)

        mask_obs = torch.zeros(
            size=[particle_num, len_seq+max_len], dtype=torch.float32, device=self.device)
        mask_unobs = torch.zeros(
            size=[particle_num, len_seq+max_len], dtype=torch.float32, device=self.device)
        mask_true = torch.ones(
            size=[particle_num], dtype=torch.float32, device=self.device )

        dtime_backward = torch.zeros(size=[particle_num, len_seq+max_len], dtype=torch.float32,
                                     device=self.device)
        index_of_hidden_backward = torch.zeros(size=[particle_num, len_seq+max_len],
                                               dtype=torch.int64, device=self.device)

        ranger = torch.arange(0, particle_num, dtype=torch.int64, device=self.device)
        current_step = torch.ones(size=[particle_num], dtype=torch.int64, device=self.device)

        self.resetPolicy(particle_num)

        if self.right2left:
            self.right2left_machine.getStatesBackwardFromSeq(one_seq[1:-1])

        types_to_update = torch.zeros(size=[particle_num], dtype=torch.int64,
                                      device=self.device)
        dtimes_to_update = torch.zeros(size=[particle_num], dtype=torch.float32,
                                       device=self.device)
        updated_with_this = torch.zeros(size=[particle_num], dtype=torch.uint8,
                                        device=self.device)

        time_so_far = torch.zeros(size=[particle_num], dtype=torch.float32,
                                  device=self.device)
        cnt = torch.zeros(size=[particle_num], dtype=torch.int64,
                          device=self.device)

        total_dtime = torch.zeros(size=[particle_num], dtype=torch.float32, device=self.device)

        dtime_bound = torch.zeros(size=[1], dtype=torch.float32, device=self.device)
        max_len = torch.empty(size=[1], dtype=torch.int64, device=self.device).fill_(max_len)

        # This variable is useful only when the resampling is on.
        log_weights = torch.zeros(
            size=[particle_num], dtype=torch.float32, device=self.device)
        # finished = torch.zeros(size=[particle_num], dtype=torch.uint8, device=self.device)
        if resampling:
            sum_log_lambda_p = torch.zeros(
                size=[particle_num], dtype=torch.float32, device=self.device)
            sum_log_lambda_q = torch.zeros(
                size=[particle_num], dtype=torch.float32, device=self.device)

            step_before_sampling = step_after_sampling = None

            def reset_resampling():
                sum_log_lambda_p.fill_(0.0)
                sum_log_lambda_q.fill_(0.0)
                log_weights.fill_(0.0)
                step_before_sampling = step_after_sampling = None

        eps = 1e-5
        total_duration = one_seq[-1]['time_since_start']

        for i_item, item in enumerate(one_seq[1:]):
            """
            sample for each interval
            in details, each x seq may have many, say I, (observed) events
            so we have I+1 intervals
            in this code, one_seq has BOS, events, EOS
            """

            if item['time_since_last_event'] / total_duration < eps:
                resampling_enabled_this_interval = False
            else:
                resampling_enabled_this_interval = True

            if resampling and resampling_enabled_this_interval:
                num_points_this_interval = num_points_for_integral[i_item]
                points_not_covered, index_not_covered, \
                total_lambda_p_atsampledpoints, \
                total_lambda_q_atsampledpoints,\
                sampled_points_integral = self.prepareResamplingInterval(
                num_points_this_interval, item, particle_num)

            dtime_bound.fill_( float(item['time_since_last_event']) )
            total_dtime.fill_(0.0)

            events_obs[:, i_item + 1] = int(item['type_event'])
            dtimes_obs[:, i_item + 1] = float(item['time_since_last_event'])

            if self.right2left:
                self.right2left_machine.prepareStatesBack(i_item)

            cnt_call = 0
            # for each item, we reset finished because:
            # even though cnt is on seq level, total_dtime is on interval level
            # so finished marker should be reset every time
            # total_dtime is larger than the dtime bound
            finished = (cnt >= max_len) | (total_dtime >= dtime_bound)

            if resampling:
                if step_before_sampling is None:
                    # used for p miss
                    # This variable is updated only when it's None
                    # Why? Because resampling is not done only when the ESS is lower than threshold
                    # If resampling is not applied at the end of an interval,
                    # p_miss should also be accumulated.
                    # Every time resampling is applied, step_before_sampling will be reset as None.
                    step_before_sampling = current_step.clone()

            while not finished.all():
                """
                keep proposing events for this interval
                until all particles are finished
                when it is finished?
                1) too many events are proposed;
                2) total dtime has exceeded dtime bound
                each propose step calls the function of sample_unobs_batch
                this function is one step of thinning algorithm
                that draw and accept an event
                although this event may go out of the current interval
                """
                """
                note that this thinning algorithm gives us
                log_lambda_p and log_lambda_q
                for likelihood computation
                this is useful for resampling
                """
                types, dtimes, log_lambda_p, log_lambda_q = self.sample_unobs_batch(
                    dtime_bound=dtime_bound, particle_num=particle_num, num=num, verbose=verbose)

                if verbose:
                    self.printDebug(cnt_call, i_item, total_dtime, dtime_bound,
                                    cnt, max_len, types, dtimes, torch.exp(log_weights))

                cnt_call += 1
                assert cnt_call <= max_len[0],\
                    "for each item, impossible to sample > {} steps, because it is the max len".format(max_len)

                if resampling and resampling_enabled_this_interval:
                    """
                    why this block needs to be here
                    because it needs the total_dtime before it is updated
                    """
                    self.computeLambdaAtSampledTimes(
                        sampled_points_integral, num_points_this_interval,
                        particle_num, dtimes, dtime_bound, total_dtime, finished,
                        total_lambda_p_atsampledpoints, total_lambda_q_atsampledpoints,
                        points_not_covered, index_not_covered)

                """
                we have to restrict the update of total_dtime:
                only unfinished will update total dtime
                because of resampling
                why? the connection may not be obvious
                if we do not have resampling, we do not mind
                adding up non-meaningful dtimes to total_dtime
                because we have dtimes_to_update which correctly tracks
                the elapsed time since last event
                and it can be directly offered by dtime (with correct indexing)
                HOWEVER, for sampled points, we have to compute
                the time diff from them to the most recent event
                for sampled points, we keep their time diff
                to the most recent observed event
                so to correctly track their time diff to most recent event
                we need to correctly compute the time diff
                between most recent event and most recent obs
                but this change should be sanity checked by seeing
                if the new code can replicate old results
                if not, maybe this change has problem
                """
                total_dtime[~finished] += dtimes[~finished]

                indices_within_interval = total_dtime < dtime_bound
                indices_not_too_many = cnt < max_len
                indices_tobe_updated = indices_within_interval & indices_not_too_many

                if resampling and resampling_enabled_this_interval:
                    """
                    why this short block has to be here?
                    because it needs indices_tobe_updated
                    """
                    # In-place operation
                    self.updateSumLogLambda(
                        sum_log_lambda_p, sum_log_lambda_q,
                        log_lambda_p, log_lambda_q,
                        indices_tobe_updated )

                types_to_update[indices_tobe_updated] = \
                types[indices_tobe_updated] #.detach()
                dtimes_to_update[indices_tobe_updated] = \
                dtimes[indices_tobe_updated] #.detach()
                updated_with_this[indices_tobe_updated] = 1

                time_so_far[indices_tobe_updated] += \
                dtimes[indices_tobe_updated] #.detach()
                cnt[indices_tobe_updated] += 1

                """
                total_logP is completely useless---even in resampling
                for resampling, we need log_pmiss, log_p and log_q
                and because of resampling,
                the final weight is always 1/M
                so what we need to output in the case of resampling
                is a vector of log weight where weight is 1/M
                """

                events_particles[ranger, current_step] = torch.where(
                    indices_tobe_updated,
                    types, events_particles[ranger, current_step]
                )
                dtimes_particles[ranger, current_step] = torch.where(
                    indices_tobe_updated,
                    dtimes, dtimes_particles[ranger, current_step]
                )

                # possibly sampled one unobserved event token
                # fill 1.0 in the right positions of mask_unobs
                mask_unobs[ranger, current_step] = torch.where(
                    indices_tobe_updated,
                    mask_true, mask_unobs[ranger, current_step]
                )

                dtime_backward[ranger, current_step] = torch.where(
                    indices_tobe_updated,
                    dtime_bound - total_dtime, dtime_backward[ranger, current_step]
                )

                temp_index = index_of_hidden_backward[ranger, current_step]
                temp_index[indices_tobe_updated] = i_item
                index_of_hidden_backward[ranger, current_step] = temp_index

                current_step[indices_tobe_updated] += 1

                if updated_with_this.any():
                    """
                    due to how updated_with_this is computed
                    particles that reach max len will not be updated
                    """
                    _ = self.update(
                        (
                            types_to_update,
                            dtimes_to_update,
                            updated_with_this ) )
                    updated_with_this.fill_(0)

                # update finished
                finished = (cnt >= max_len) | (total_dtime >= dtime_bound)
            # END while

            """
            the code block below should be here
            it should be at after the interval is finished
            because it checks all the non-covered time points
            """
            if resampling and resampling_enabled_this_interval:
                integral_p, integral_q = self.computeIntegralInterval(
                    points_not_covered, index_not_covered,
                    sampled_points_integral, total_dtime, dtime_bound,
                    num_points_this_interval, particle_num,
                    total_lambda_p_atsampledpoints,
                    total_lambda_q_atsampledpoints )

            # update with observed events

            type_event_obs = int(item['type_event'])
            total_time_obs = float(item['time_since_start'])

            dtimes_to_update = total_time_obs - time_so_far
            types_to_update.fill_(type_event_obs)
            updated_with_this.fill_(1)

            events_particles[ranger, current_step] = type_event_obs#.detach()
            dtimes_particles[ranger, current_step] = dtimes_to_update#.detach()

            if i_item < len(one_seq) - 2:
                mask_obs[ranger, current_step] = mask_true

            current_step[:] += 1

            if resampling:
                # used for p miss
                # I put it here because I need the "plus 1".
                step_after_sampling = current_step.clone()

            time_so_far.fill_(total_time_obs)
            hy_forward_before_obs = self.update(
                (
                    types_to_update,
                    dtimes_to_update,
                    updated_with_this))
            updated_with_this.fill_(0)

            if resampling and resampling_enabled_this_interval:
                log_pmiss = self.miss_mec.clip_probability(step_before_sampling, step_after_sampling,
                                                           mask_obs, events_particles, dtimes_particles)

                # All the variables needed to be updated for resampling
                # Here I just list all the variables whose 1st dimension is particle_num, and comment out
                # the ones that obviously do not need to be resampled.
                # There must be more redundant variables below, but I think it's OKay to keep them here.
                # They are all local variables, and it does no harm if we do resampling for them.
                # For efficiency, we may need to get rid of them.
                resampling_variables = [
                    current_step,
                    cnt,
                    dtime_backward,
                    # dtimes,
                    dtimes_obs,
                    dtimes_particles,
                    # dtimes_to_update,
                    duration,
                    events_obs,
                    events_particles,
                    # finished,
                    hy_forward_before_obs,
                    # index_not_covered,
                    index_of_hidden_backward,
                    # indices_not_too_many,
                    # indices_tobe_updated,
                    # indices_within_interval,
                    # integral_p,
                    # integral_q,
                    # log_lambda_p,
                    # log_lambda_q,
                    # log_pmiss,
                    # log_weights,
                    mask_obs,
                    mask_true,
                    mask_unobs,
                    # points_not_covered,
                    # ranger, # DO NOT INCLUDE THIS! This variable has nothing to do with particles.
                    # step_after_sampling,
                    # step_before_sampling,
                    # sum_log_lambda_p,
                    # sum_log_lambda_q,
                    # temp_index,
                    time_so_far,
                    # total_dtime,
                    # total_lambda_p_atsampledpoints,
                    # total_lambda_q_atsampledpoints,
                    # types,
                    types_to_update,
                    updated_with_this,

                    *self.states
                ]

                sum_log_lambda_p = self.computeWeight(
                    i_item, len_seq,
                    hy_forward_before_obs, type_event_obs,
                    sum_log_lambda_p, sum_log_lambda_q, integral_p, integral_q,
                    log_pmiss, log_weights)

                if need_eliminate_log_base:
                    log_weights = self.eliminate_log_base(log_weights)
                if self.need_resampling_now(log_weights, resampling_threshold_ratio, particle_num):
                    self.doMultinomialResampling(log_weights, resampling_variables)
                    # Reset the sum_log_lambda's and log_weights
                    reset_resampling()

            # finish if resampling
        # finish looping over obs seq

        lens = cnt + len_seq - 2 # total # of obs + unobs events
        max_of_lens = lens.max()

        # trim extra useless padding
        events_particles = events_particles[:, :max_of_lens + 2 ]
        dtimes_particles = dtimes_particles[:, :max_of_lens + 2 ]

        mask_obs = mask_obs[:, :max_of_lens + 2]
        mask_unobs = mask_unobs[:, :max_of_lens + 2]

        dtime_backward = dtime_backward[:, :max_of_lens + 2 ]
        index_of_hidden_backward = index_of_hidden_backward[:, :max_of_lens + 2 ]
        r"""
        Note that we need to operate the index_of_hidden_backward
        so it can index the right positions when the indexed tensor is flatten
        the operation is
        for i-th row of the matrix, add all elements by (len_seq - 1) * particle_num
        """
        indices_mat = torch.arange(0, particle_num, dtype=torch.int64,
            device=self.device).unsqueeze(1).expand(particle_num, max_of_lens+2)
        indices_mat = (len_seq - 1) * indices_mat
        index_of_hidden_backward += indices_mat
        r"""
        output interface should exactly match that of processSeq
        precisely
        processSeq : make synthetic x and z given ONE complete seq
        this method : propose z_m m=1,...,M given ONE observed seq
        so particle_num in this method == n in that method of processSeq
        """

        # We should return the weights of particles if we do resampling.
        # We do so since at some cases the weights of particles are not all 1's.
        # E.g. If the resampling is done adaptively, there's a chance that there's no
        # resampling at the last step, thus the weights of particles are different.
        if need_eliminate_log_base:
            log_weights = self.eliminate_log_base(log_weights)
        return [
            events_particles, # 0: particle_num * len_complete
            dtimes_particles, # 1: particle_num * len_complete
            log_weights, # 2: particle_num # If resampling is off, this variable is meaningless.
            duration, # 3: particle_num
            lens, # 4: particle_num # lens of proposed complete seqs (without BOS EOS)
            events_obs, # 5: particle_num * len_obs
            dtimes_obs, # 6: particle_num * len_obs
            dtime_backward, # 7: particle_num * len_complete
            index_of_hidden_backward, # 8: particle_num * len_complete
            mask_obs, # 9: particle_num * len_complete # 1.0 for observed events all others including BOS EOS PAD are all 0.0
            mask_unobs # 10: particle_num * len_complete # 1.0 for unobserved events all others including BOS EOS PAD are all 0.0
            # log_censor_probs # 11: need to be filled by missing mec
            ]

    @staticmethod
    def eliminate_log_base(vector):
        """
        Vectors may be processed in log space. When we translate them into the normal space,
        they may go beyond the precision of double float. Sometimes we may only care about the
        relative value of there vectors. E.g., when we need to normalize the vector, [-314.2, -314.0]
        is equivalent to [-0.2, 0.0]. In these situations, we could eliminate the "base" of the vector
        to avoid the precision overflow.
        :param torch.Tensor vector: Input vector.
        :return: The tensor with base eliminated.
        """
        return vector - torch.max(vector)
