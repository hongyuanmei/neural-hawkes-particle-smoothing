import torch
import torch.nn as nn
import torch.nn.functional as F


class CTLSTMCell(nn.Module):

    def __init__(self, hidden_dim, beta=1.0, device=None):
        super(CTLSTMCell, self).__init__()

        device = device or 'cpu'
        self.device = torch.device(device)

        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(hidden_dim * 2, hidden_dim * 7, bias=True)

        self.beta = beta

    def forward(
            self, rnn_input,
            hidden_t_i_minus, cell_t_i_minus, cell_bar_im1):

        dim_of_hidden = rnn_input.dim() - 1

        input_i = torch.cat((rnn_input, hidden_t_i_minus), dim=dim_of_hidden)
        output_i = self.linear(input_i)

        gate_input, \
        gate_forget, gate_output, gate_pre_c, \
        gate_input_bar, gate_forget_bar, gate_decay = output_i.chunk(
            7, dim_of_hidden)

        gate_input = torch.sigmoid(gate_input)
        gate_forget = torch.sigmoid(gate_forget)
        gate_output = torch.sigmoid(gate_output)
        gate_pre_c = torch.tanh(gate_pre_c)
        gate_input_bar = torch.sigmoid(gate_input_bar)
        gate_forget_bar = torch.sigmoid(gate_forget_bar)
        gate_decay = F.softplus(gate_decay, beta=self.beta)

        cell_i = gate_forget * cell_t_i_minus + gate_input * gate_pre_c
        cell_bar_i = gate_forget_bar * cell_bar_im1 + gate_input_bar * gate_pre_c

        return cell_i, cell_bar_i, gate_decay, gate_output

    def decay(self, cell_i, cell_bar_i, gate_decay, gate_output, dtime):
        # no need to consider extra_dim_particle here
        # cuz this function is applicable to any # of dims
        if dtime.dim() < cell_i.dim():
            dtime = dtime.unsqueeze(cell_i.dim()-1).expand_as(cell_i)

        cell_t_ip1_minus = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(
            -gate_decay * dtime)
        hidden_t_ip1_minus = gate_output * torch.tanh(cell_t_ip1_minus)

        return cell_t_ip1_minus, hidden_t_ip1_minus
