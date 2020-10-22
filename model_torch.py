import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable






# 超参数个数：5
hparams = {
    'phn_num': 345,
    'n_mels' : 80,
    'spec_dim' : 201,

    'hidden_units' : 256,
    'hidden_dim_before_second_cbhg_div2' : 256,
}

use_cuda = torch.cuda.is_available()



class SeqLinear(nn.Module):
    """
    Linear layer for sequences
    """
    def __init__(self, input_size, output_size, time_dim=1):
        """
        :param input_size: dimension of input
        :param output_size: dimension of output
        :param time_dim: index of time dimension
        """
        super(SeqLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_dim = time_dim
        self.linear = nn.Linear(input_size, output_size)


    def forward(self, input_):
        """
        :param input_: sequences
        :return: outputs
        """
        batch_size = input_.size()[0]
        if self.time_dim == 2:
            input_ = input_.transpose(1, 2).contiguous()
        input_ = input_.view(-1, self.input_size)

        out = self.linear(input_).view(batch_size, -1, self.output_size)

        if self.time_dim == 2:
            out = out.contiguous().transpose(1, 2)

        return out


class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', SeqLinear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(0.5)),
             ('fc2', SeqLinear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(0.5)),
        ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out


class Highwaynet(nn.Module):
    """
    Highway network
    """
    def __init__(self, num_units, num_layers=4):
        """
        :param num_units: dimension of hidden unit
        :param num_layers: # of highway layers
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(SeqLinear(num_units, num_units))
            self.gates.append(SeqLinear(num_units, num_units))

    def forward(self, input_):

        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):

            h = F.relu(fc1.forward(out))
            t = F.sigmoid(fc2.forward(out))

            c = 1. - t
            out = h * t + out * c

        return out


class CBHG(nn.Module):
    """
    CBHG Module
    """
    def __init__(self, hidden_size, K=16, projection_size = 128, num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
        """
        :param hidden_size: dimension of hidden unit
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        :param is_post: whether post processing or not
        """
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(nn.Conv1d(in_channels=projection_size,
                                                out_channels=hidden_size,
                                                kernel_size=1,
                                                padding=int(np.floor(1/2))))

        for i in range(2, K+1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                                                out_channels=hidden_size,
                                                kernel_size=i,
                                                padding=int(np.floor(i/2))))

        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K+1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K
        if is_post:
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size * 2,
                                             kernel_size=3,
                                             padding=int(np.floor(3/2)))
            self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size * 2,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3/2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size * 2)

        else:
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size,
                                             kernel_size=3,
                                             padding=int(np.floor(3 / 2)))
            self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3 / 2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)


        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size, num_layers=2,
                          batch_first=True,
                          bidirectional=True)


    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:,:,:-1]
        else:
            return x

    def forward(self, input_):

        input_ = input_.contiguous()
        # print("INNER CBHG input_ 1:", input_.size())
        batch_size = input_.size()[0]

        convbank_list = list()
        input_ = input_.transpose(1, 2).contiguous()
        # print("INNER CBHG input_ 2:", input_.size())
        convbank_input = input_

        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = F.relu(batchnorm(self._conv_fit_dim(conv(convbank_input), k+1).contiguous()))
            convbank_list.append(convbank_input)

        # Concatenate all features
        # for chech_cnn_out in convbank_list:
        #     print("INNER convbank_list:", chech_cnn_out.size())
        conv_cat = torch.cat(convbank_list, dim=1)
        # print("INNER conv_cat:", conv_cat.size())

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:,:,:-1]

        # Projection
        conv_projection = F.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection))) + input_

        # print("INNER conv_projection 1:", conv_projection.size())
        conv_projection = conv_projection.transpose(1, 2).contiguous()
        # print("INNER conv_projection 2:", conv_projection.size())

        # Highway networks
        highway = self.highway.forward(conv_projection)
        # highway = torch.transpose(highway, 1,2)

        # Bidirectional GRU
        if use_cuda:
            init_gru = Variable(torch.zeros(2 * self.num_gru_layers, batch_size, self.hidden_size)).cuda()
        else:
            init_gru = Variable(torch.zeros(2 * self.num_gru_layers, batch_size, self.hidden_size))

        self.gru.flatten_parameters()
        out, _ = self.gru(highway, init_gru)

        return out


class DCBHG(nn.Module):
  def __init__(self):
    super(DCBHG, self).__init__()
    self.input_dim_phn = hparams['phn_num']
    # self.hidden_dim_before_second_cbhg_div2 = hparams['hidden_dim_before_second_cbhg_div2']

    self.prenet = Prenet(self.input_dim_phn, hparams['hidden_units'], hparams['hidden_units'] // 2)
    self.cbhg1 = CBHG(hparams['hidden_units'])
    self.linear1 = nn.Linear(hparams['hidden_units'] * 2, hparams['n_mels'])

    self.linear2 = nn.Linear(hparams['n_mels'], hparams['hidden_dim_before_second_cbhg_div2'] // 2)
    self.cbhg2 = CBHG(hparams['hidden_units'])
    self.linear3 = nn.Linear(hparams['hidden_units'] * 2, hparams['spec_dim'])



  def forward(self, ppgs):
    prenet_out = self.prenet(ppgs)
    pred_mel = self.cbhg1(prenet_out)
    pred_mel = self.linear1(pred_mel)

    pred_spec = self.linear2(pred_mel)
    pred_spec = self.cbhg2(pred_spec)
    pred_spec = self.linear3(pred_spec)

    return pred_mel, pred_spec