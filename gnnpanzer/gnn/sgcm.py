import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import utils

from typing import Union, Optional, Callable, Optional

from tqdm import tqdm
from torch_geometric.nn.conv import SGConv

from utils import Arrays


class SGCModel(nn.Module):
    def __init__(self, in_channels: int, k: int, out_channels: int,
                 dropout: float = 0.5, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False, add_self_loops=True, conv_cached=False):
        super().__init__()
        self.nlayers = 1
        # init conv
        # not use cache for evasion setting or inductive setting
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lns = nn.ModuleList()

        self.convs.append(SGConv(in_channels, out_channels, k, conv_cached, add_self_loops, with_bias))

        if with_bn:
            warnings.warn('BatchNorm not available.')
        if with_ln:
            self.lns.append(nn.LayerNorm(in_channels))

        # control
        self.conv_cached = conv_cached
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.with_softmax = with_softmax
        self.with_bn = with_bn
        self.with_ln = with_ln  # default layer norm first
        # fit
        self.dropout = dropout
        # self.lr = lr
        # self.weight_decay = wd

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def clear_conv_cache(self):
        if self.conv_cached:
            for conv in self.convs:
                conv._cached_x = None

    def forward(self, x, edge_index):
        # not support edge_weight and edge_attr yet
        if self.with_ln:
            x = self.lns[0](x)

        x = self.convs[0](x, edge_index)

        # if self.with_relu:
        #     x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)

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
