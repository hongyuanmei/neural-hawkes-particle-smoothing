import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nhps.models.cont_time_cell import CTLSTMCell
from nhps.models.utils import makeHiddenForBound, makeHiddenForLambda


class Right2Left(nn.Module):
    def __init__(self, hidden_dim, beta, total_num, device, type_back, l2r_LSTM_dim):
        super(Right2Left, self).__init__()
        device = device or 'cpu'
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.total_num = total_num
        self.type_back = type_back

        self.idx_BOS = self.total_num
        self.idx_EOS = self.total_num + 1
        self.idx_PAD = self.total_num + 2

        self.Emb = nn.Embedding(
            self.total_num + 3, self.hidden_dim)

        self.rnn_cell = CTLSTMCell(
            self.hidden_dim, beta=self.beta, device=self.device)

        self.init_h = torch.zeros(size=[hidden_dim],
                                  dtype=torch.float32, device=self.device)
        self.init_c = torch.zeros(size=[hidden_dim],
                                  dtype=torch.float32, device=self.device)
        self.init_cb = torch.zeros(size=[hidden_dim],
                                   dtype=torch.float32, device=self.device)
        self.eps = np.finfo(float).eps
        self.max = np.finfo(float).max

        self.list_states_back = list()
        self.states_back = None

        if self.type_back == 'sep':
            r"""
            \lambda^c_k = softplus( w_k h + v_k h_back )
            """
            self.hidden_lambda_back = nn.Linear(
                self.hidden_dim, self.total_num, bias=False )
        elif self.type_back == 'add':
            r"""
            \lambda^c_k = softplus( w_k ( h + V h_back ) )
            \lambda^c_k = softplus( w_k h + w_k V h_back )
            """
            self.hidden_lambda_back = nn.Linear(
                self.hidden_dim, l2r_LSTM_dim, bias=False)
        elif self.type_back == 'mul':
            r"""
            \lambda^c_k = \lambda_k \lambda_k_back
            \lambda^c_k = softplus( w_k h ) softplus( v_k [h, h_back] )
            """
            self.hidden_lambda_back = nn.Linear(
                l2r_LSTM_dim + self.hidden_dim, self.total_num, bias=False)
        else:
            raise Exception(
                'Unknown backward machine type : {}'.format(self.type_back))

    def forward(self, event_obs, dtime_obs):
        r"""
        go through the sequences and get all states from RIGHT to LEFT
        this function is ONLY used when computing \sum_j q(z_j | H_j, F_j)
        and it ONLY computes the F_j part
        this seq is SAME FOR ALL the particles of it
        """
        r"""
        note that the event_obs and dtime_obs is still sorted in temporal order
        e.g. when type-0 is observed and type-1 is always missing
        (so idx_BOS=2, idx_EOS=3, idx_PAD=4)
        event_obs may be
        2 0 1 0 1 3 4 4
        2 0 1 0 1 0 1 3
        so we should :
        1. process them in reverse order to get right2left states
        2. mask states to avoid the effect of PAD
        """
        r"""
        Note the way event_obs and dtime_obs match is
        the SAME with how event and dtime match --- each event with its dtime
        so when decaying hidden states, we should access
        dtime_obs[:, :, i] ( NOT like dtime, we access [:, :, i+1 ] )
        """
        batch_size, num_particles, T_obs_plus_2 = event_obs.size()
        cell_t_i_plus = self.init_c.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.hidden_dim)
        cell_bar_ip1 = self.init_cb.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.hidden_dim)
        hidden_t_i_plus = self.init_h.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.hidden_dim)

        all_cell, all_cell_bar = [], []
        all_gate_output, all_gate_decay = [], []
        all_hidden = []
        all_hidden_after_update = []

        for j in range(T_obs_plus_2 - 1):
            # only EOS to first event update LSTM

            # <s> backward CT-LSTM
            i = T_obs_plus_2 - 1 - j

            event_i = event_obs[:, :, i ]
            emb_i = self.Emb( event_i )
            dtime_i = dtime_obs[:, :, i ]

            cell_i, cell_bar_i, gate_decay_i, gate_output_i = self.rnn_cell(
                emb_i, hidden_t_i_plus, cell_t_i_plus, cell_bar_ip1
            )
            _, hidden_t_i_minus = self.rnn_cell.decay(
                cell_i, cell_bar_i, gate_decay_i, gate_output_i,
                torch.zeros_like(dtime_i)
            )
            cell_t_im1_plus, hidden_t_im1_plus = self.rnn_cell.decay(
                cell_i, cell_bar_i, gate_decay_i, gate_output_i,
                dtime_i
            )

            # mask the cell and hidden if this step is PAD
            # PAD does not update init states
            mask_i = torch.ones_like(dtime_i)
            mask_i[ event_i == self.idx_PAD ] = 0.0
            mask_i = mask_i.unsqueeze(2)
            cell_t_im1_plus = cell_t_im1_plus * mask_i
            cell_bar_i = cell_bar_i * mask_i
            hidden_t_im1_plus = hidden_t_im1_plus * mask_i

            all_cell.append(cell_i)
            all_cell_bar.append(cell_bar_i)
            all_gate_decay.append(gate_decay_i)
            all_gate_output.append(gate_output_i)
            all_hidden.append(hidden_t_im1_plus)
            all_hidden_after_update.append(hidden_t_i_minus)

            cell_t_i_plus = cell_t_im1_plus
            cell_bar_ip1 = cell_bar_i
            hidden_t_i_plus = hidden_t_im1_plus
            # </s> backward CT-LSTM

        all_cell = torch.stack( all_cell[::-1], dim=2 )
        all_cell_bar = torch.stack( all_cell_bar[::-1], dim=2 )
        all_gate_decay = torch.stack( all_gate_decay[::-1], dim=2 )
        all_gate_output = torch.stack( all_gate_output[::-1], dim=2 )
        all_hidden = torch.stack( all_hidden[::-1], dim=2 )
        all_hidden_after_update = torch.stack( all_hidden_after_update[::-1], dim=2 )
        # only EOS to first event update LSTM
        # so these states start from the 1st event (with index 0)

        return T_obs_plus_2, all_cell, all_cell_bar, all_gate_decay, \
               all_gate_output, all_hidden, all_hidden_after_update

    def cuda(self, device=None):
        device = device or 'cuda:0'
        self.device = torch.device(device)
        assert self.device.type == 'cuda'
        super().cuda(self.device)

    def cpu(self):
        self.device = torch.device('cpu')
        super().cuda(self.device)

    def getLambdaCorrected(self,
                           hy_forward, hy_forward_sample,
                           dtime_backward, index_of_hidden_backward,
                           dtime_backward_sampling, index_of_hidden_backward_sampling,
                           target, mask_unobs,
                           all_cell_obs, all_cell_bar_obs, all_gate_output_obs, all_gate_decay_obs,
                           hidden_lambda):

        r"""
        Get lambda_corrected, i.e., intensities used for proposing
        It is conditioned on
        BOTH the past and future, which is particle smoothing
        Note that for smoothing, needed are the hidden states from both
        left-to-right and right-to-left machine
        and need to index the correct states for each event
        Also note that this method computes intensities at
        both occurrence and sampled times
        """
        #print("dtime_backward shape : {}".format(dtime_backward.size()))

        batch_size, num_particles, T_plus_2 = dtime_backward.size()
        _, _, T_obs_plus_1, _ = all_cell_obs.size()
        _, _, max_len_sampling = dtime_backward_sampling.size()
        T_plus_1 = T_plus_2 - 1
        r"""
        dtime_backward, index_of_hidden_backward :
        batch_size * num_particles * T+2
        dtime_backward : time upto the earliest observed event in future
        index_of_hidden_backward : index of states of earliest observed in future
        all_*_obs : batch_size * 1 * T_obs+1 * hidden_dim
        """

        index_of_hidden_backward_notfor_BOS = index_of_hidden_backward[:,:,1:].contiguous()

        #<s> log lambda for events

        all_cell_back = all_cell_obs.view(
            batch_size * num_particles * T_obs_plus_1, self.hidden_dim )[
                        index_of_hidden_backward_notfor_BOS.view(-1), :].view(
            batch_size, num_particles, T_plus_1, self.hidden_dim )

        all_cell_bar_back = all_cell_bar_obs.view(
            batch_size * num_particles * T_obs_plus_1, self.hidden_dim )[
                            index_of_hidden_backward_notfor_BOS.view(-1), :].view(
            batch_size, num_particles, T_plus_1, self.hidden_dim )
        all_gate_output_back = all_gate_output_obs.view(
            batch_size * num_particles * T_obs_plus_1, self.hidden_dim )[
                               index_of_hidden_backward_notfor_BOS.view(-1), :].view(
            batch_size, num_particles, T_plus_1, self.hidden_dim )
        all_gate_decay_back = all_gate_decay_obs.view(
            batch_size * num_particles * T_obs_plus_1, self.hidden_dim )[
                              index_of_hidden_backward_notfor_BOS.view(-1), :].view(
            batch_size, num_particles, T_plus_1, self.hidden_dim )

        cy_back, hy_back = self.rnn_cell.decay(
            all_cell_back, all_cell_bar_back,
            all_gate_decay_back, all_gate_output_back,
            dtime_backward[:, :, 1:]
        )

        all_lambda_corrected = self.helpGetLambdaCorrected(
            hy_forward, hy_back, hidden_lambda)

        log_lambda_corrected = torch.log(all_lambda_corrected + self.eps)
        log_lambda_corrected_target = log_lambda_corrected.view(
            batch_size * num_particles * T_plus_1, self.total_num
        )[
            torch.arange(0, batch_size * num_particles * T_plus_1,
                         dtype=torch.int64, device=self.device),
            target.view( batch_size * num_particles * T_plus_1 )
        ].view(batch_size, num_particles, T_plus_1)

        log_lambda_corrected_target_unobs = log_lambda_corrected_target * mask_unobs

        lambda_corrected_sum_unobs = torch.sum(all_lambda_corrected, dim=3)
        log_lambda_corrected_sum_unobs = torch.log(lambda_corrected_sum_unobs + self.eps)
        log_lambda_corrected_sum_unobs *= mask_unobs
        #</s>

        #<s> lambda for sampled times
        all_cell_back_sampling = all_cell_obs.view(
            batch_size * num_particles * T_obs_plus_1, self.hidden_dim )[
                                 index_of_hidden_backward_sampling.view(-1), :].view(
            batch_size, num_particles, max_len_sampling, self.hidden_dim)
        all_cell_bar_back_sampling = all_cell_bar_obs.view(
            batch_size * num_particles * T_obs_plus_1, self.hidden_dim )[
                                     index_of_hidden_backward_sampling.view(-1), :].view(
            batch_size, num_particles, max_len_sampling, self.hidden_dim)
        all_gate_output_back_sampling = all_gate_output_obs.view(
            batch_size * num_particles * T_obs_plus_1, self.hidden_dim )[
                                        index_of_hidden_backward_sampling.view(-1), :].view(
            batch_size, num_particles, max_len_sampling, self.hidden_dim)
        all_gate_decay_back_sampling = all_gate_decay_obs.view(
            batch_size * num_particles * T_obs_plus_1, self.hidden_dim )[
                                       index_of_hidden_backward_sampling.view(-1), :].view(
            batch_size, num_particles, max_len_sampling, self.hidden_dim)

        cy_back_sample, hy_back_sample = self.rnn_cell.decay(
            all_cell_back_sampling, all_cell_bar_back_sampling,
            all_gate_decay_back_sampling, all_gate_output_back_sampling,
            dtime_backward_sampling
        )

        all_lambda_corrected_sample = self.helpGetLambdaCorrected(
            hy_forward_sample, hy_back_sample, hidden_lambda)

        #</s>
        return log_lambda_corrected_target_unobs, all_lambda_corrected_sample

    def helpGetLambdaCorrected(self, hy_forward, hy_back, hidden_lambda):

        if self.type_back == 'sep':

            lambda_corrected = F.softplus(
                hidden_lambda(hy_forward) + self.hidden_lambda_back(hy_back),
                beta=self.beta )

        elif self.type_back == 'add':

            lambda_corrected = F.softplus(
                hidden_lambda(
                    hy_forward + self.hidden_lambda_back(hy_back) ), beta=self.beta )

        elif self.type_back == 'mul':

            lambda_corrected = F.softplus(
                hidden_lambda(hy_forward), beta=self.beta
            ) * F.softplus(
                self.hidden_lambda_back(
                    torch.cat( (hy_forward, hy_back), dim=3 )
                ), beta=1.0
            )

        else:

            raise Exception(
                "Unknown backward machine type : {}".format(self.type_back) )

        return lambda_corrected

    def getStatesBackwardFromSeq(self, one_seq):
        r"""
        make event and dtime and then call getStatesBackward
        """
        len_seq = len(one_seq)
        event_obs = torch.zeros(size=[1, 1, len_seq+2], dtype=torch.int64, device=self.device)
        event_obs[:, :, 0] = self.idx_BOS
        event_obs[:, :, len_seq + 1] = self.idx_EOS
        dtime_obs = torch.zeros(size=[1, 1, len_seq+2], dtype=torch.float32, device=self.device)
        for i_item, item in enumerate(one_seq):
            event_obs[:, :, i_item + 1] = int(item['type_event'])
            dtime_obs[:, :, i_item + 1] = float(item['time_since_last_event'])
        output = self(event_obs, dtime_obs)
        self.list_states_back = list()
        for i_item in range(len_seq + 1):
            self.list_states_back.append(
                (
                    output[1][0, :, i_item], output[2][0, :, i_item],
                    output[3][0, :, i_item], output[4][0, :, i_item]
                )
            )

    def prepareStatesBack(self, i_item):
        self.states_back = self.list_states_back[i_item]

    def pack_bound(self, hidden_lambda, states, dtime, particle_num,
                   dtime_back, projected_hidden
                   ):
        if self.type_back == 'sep':

            linear_weight_back = self.hidden_lambda_back.weight.transpose(0, 1)
            hy_backward = makeHiddenForBound(linear_weight_back, self.states_back,
                                             dtime_back, particle_num,
                                             self.hidden_dim, self.total_num)
            linear_weight_back = linear_weight_back.unsqueeze(dim=0).expand(
                particle_num, *linear_weight_back.size())
            projected_hidden_back = (hy_backward * linear_weight_back).sum(1)
            projected_hidden_both = projected_hidden + projected_hidden_back
            lambda_bound = F.softplus(projected_hidden_both, beta=self.beta)

        elif self.type_back == 'add':
            r"""
            NOTE : the parametrization seems different in different places for 'add'
            But it is actually the SAME (and of course correct)!
            To get the upper bound for \lambda_k
            we need to rewrite
            w_k (h + V h_back) = w_k h + (w_k V) h_back and
            treat w_k V as a new linear weight
            so that we can adjust elements of h_back accordingly
            """
            linear_weight_back = torch.mm(
                self.hidden_lambda_back.weight.transpose(0, 1),
                hidden_lambda.weight.transpose(0, 1) )
            hy_backward = makeHiddenForBound(linear_weight_back, self.states_back, dtime_back,
                                             particle_num, self.hidden_dim, self.total_num)
            linear_weight_back = linear_weight_back.unsqueeze(dim=0).expand(
                particle_num, *linear_weight_back.size())
            projected_hidden_back = (hy_backward * linear_weight_back).sum(1)
            projected_hidden_both = projected_hidden + projected_hidden_back
            lambda_bound = F.softplus(projected_hidden_both, beta=self.beta)

        elif self.type_back == 'mul':

            linear_weight_back = self.hidden_lambda_back.weight.transpose(0, 1)
            hy_forward_for_back = makeHiddenForBound(linear_weight_back[:self.hidden_dim, :],
                                                     states, dtime, particle_num, self.hidden_dim,
                                                     self.total_num)
            hy_backward = makeHiddenForBound(
                linear_weight_back[self.hidden_dim:, :], self.states_back, dtime_back,
                particle_num, self.hidden_dim, self.total_num
            )
            linear_weight_back = linear_weight_back.unsqueeze(dim=0).expand(
                particle_num, *linear_weight_back.size() )
            projected_hidden_back = (
                    torch.cat(
                        (hy_forward_for_back, hy_backward), dim=1
                    ) * linear_weight_back ).sum(1)
            lambda_bound = F.softplus(
                projected_hidden, beta=self.beta) * F.softplus(
                projected_hidden_back, beta=1.0)

        else:

            raise Exception(
                "Unknown backward machine type : {}".format(self.type_back))

        return lambda_bound

    def pack_lambda(self, hidden_lambda,
                    dtime_back, hy_forward, projected_hidden):

        hy_backward = makeHiddenForLambda(
            self.states_back, dtime_back)

        if self.type_back == 'sep':

            projected_hidden += self.hidden_lambda_back(hy_backward)
            lambda_t = F.softplus( projected_hidden, beta=self.beta )

        elif self.type_back == 'add':

            projected_hidden += hidden_lambda(
                self.hidden_lambda_back(hy_backward) )
            lambda_t = F.softplus( projected_hidden, beta=self.beta )

        elif self.type_back == 'mul':

            projected_hidden_back = self.hidden_lambda_back(
                torch.cat(
                    (hy_forward, hy_backward), dim=2) )
            lambda_t = F.softplus(
                projected_hidden, beta=self.beta ) * F.softplus(
                projected_hidden_back, beta=1.0)

        else:

            raise Exception(
                "Unknown backward machine type : {}".format(self.type_back))

        return lambda_t
