

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import utils

from typing import Union, Optional, Callable, Optional

from tqdm import tqdm
from torch_geometric.nn.conv import GATConv

from utils import Arrays


class GATModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Union[int, Arrays], num_layers: int, out_channels: int, heads: Union[int, Arrays],
                 dropout: float = 0.5, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False,
                 add_self_loops=True, att_concat=True, ns=0.2, att_dropout: float = 0):
        super().__init__()
        self.nlayers = num_layers
        # init conv
        # not use cache for evasion setting or inductive setting
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lns = nn.ModuleList()

        # hidden dim list
        if type(hidden_channels) == int:
            hidden_channels = (np.ones(num_layers-1) * hidden_channels).astype(int)
        else:
            assert len(hidden_channels) == num_layers-1, 'dim not match'
        if type(heads) == int:
            heads = (np.ones(num_layers-1) * heads).astype(int)
        else:
            assert len(heads) == num_layers - 1, 'heads not match'
        if not att_concat:
            raise NotImplementedError('not concat gat heads')
        # input
        self.convs.append(GATConv(in_channels, hidden_channels[0], heads[0], concat=True,
                                  bias=with_bias, add_self_loops=add_self_loops, negative_slope=ns, dropout=att_dropout))
        if with_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels[0] * heads[0]))
        if with_ln:
            self.lns.append(nn.LayerNorm(in_channels))
        # middle_hidden
        for layer_i in range(num_layers-2):
            self.convs.append(GATConv(int(hidden_channels[layer_i] * heads[layer_i]), hidden_channels[layer_i+1], heads[layer_i+1],
                                      concat=True, bias=with_bias, add_self_loops=add_self_loops,
                                      negative_slope=ns, dropout=att_dropout))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels[layer_i+1] * heads[layer_i+1]))
            if with_ln:
                self.lns.append(nn.LayerNorm(hidden_channels[layer_i] * heads[layer_i]))
        # output
        if with_ln:
            self.lns.append(nn.LayerNorm(hidden_channels[-1] * heads[-1]))
        self.convs.append(GATConv(int(hidden_channels[-1] * heads[-1]), out_channels, 1, concat=False,
                                  bias=with_bias, add_self_loops=add_self_loops, negative_slope=ns, dropout=att_dropout))
        # single layer
        # self.convs.append(GCNConv(in_channels, out_channels,
        #                           bias=with_bias, normalize=normalize, add_self_loops=add_self_loops, cached=conv_cached))
        # control
        self.conv_cached = False
        self.with_relu = with_relu  # 注意 gatconv 自带 leaky_relu 用于注意力分数计算
        self.with_bias = with_bias
        self.with_softmax = with_softmax
        self.with_bn = with_bn
        self.with_ln = with_ln  # default layer norm first
        # fit
        self.dropout = dropout  # 注意 gatconv 自带 dropout 用于注意力分数计算
        # self.lr = lr
        # self.weight_decay = wd

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x, edge_index):
        # not support edge_weight and edge_attr yet
        for i in range(self.nlayers):
            if self.with_ln:
                x = self.lns[i](x)
            x = self.convs[i](x, edge_index)

            if i == self.nlayers - 1:
                break

            if self.with_bn:
                x = self.bns[i](x)

            if self.with_relu:
                x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        if self.with_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

    @torch.no_grad()
    def get_linear_weight(self):
        """获取线性化的权重矩阵，探测模型中的二维矩阵并相乘"""
        weight = None
        for para in self.parameters():
            if para.ndim == 1:
                continue
            if weight is None:
                weight = para.detach().T
            else:
                weight = weight @ para.detach().T
        return weight
