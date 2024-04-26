import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch_scatter

import utils

from typing import Union, Optional, Callable, Optional

from tqdm import tqdm
from torch_geometric.nn.conv import GCNConv
from torch_sparse import SparseTensor

from utils import Arrays


class GCNGuard(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Union[int, Arrays], num_layers: int, out_channels: int,
                 dropout: float = 0.0, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False,
                 normalize=True, add_self_loops=True, conv_cached=False,
                 with_gate=True, threshold=0.1):
        super().__init__()
        self.nlayers = num_layers
        # init conv
        # not use cache for evasion setting
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lns = nn.ModuleList()
        if type(hidden_channels) == int:
            hidden_channels = (np.ones(num_layers-1) * hidden_channels).astype(int)
        else:
            assert len(hidden_channels) == num_layers-1, 'dim not match'
        if conv_cached:
            conv_cached = False
            warnings.warn('gcnguard not support gcn conv cache')
        # input
        self.convs.append(GCNConv(in_channels, hidden_channels[0],
                                  bias=with_bias, normalize=normalize, add_self_loops=add_self_loops))
        if with_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels[0]))
        if with_ln:
            self.lns.append(nn.LayerNorm(in_channels))
        # middle_hidden
        for layer_i in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels[layer_i], hidden_channels[layer_i+1],
                                      bias=with_bias, normalize=normalize, add_self_loops=add_self_loops))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels[layer_i + 1]))
            if with_ln:
                self.lns.append(nn.LayerNorm(hidden_channels[layer_i]))
        # output
        if with_ln:
            self.lns.append(nn.LayerNorm(hidden_channels[-1]))
        self.convs.append(GCNConv(hidden_channels[-1], out_channels,
                                  bias=with_bias, normalize=normalize, add_self_loops=add_self_loops))
        # single layer
        # self.convs.append(GCNConv(in_channels, out_channels,
        #                           bias=with_bias, normalize=normalize, add_self_loops=add_self_loops))
        self.with_gate = with_gate
        self.gate = nn.Parameter(torch.rand(1))
        self.threshold = threshold
        self.add_self_loops = add_self_loops
        # control
        self.conv_cached = conv_cached
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.with_softmax = with_softmax
        self.with_bn = with_bn
        self.with_ln = with_ln  # default layer norm first
        # fit
        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def get_attention_coefficient(self, x, edge_index, add_self_loop):
        row, col = edge_index[0], edge_index[1]
        if x.is_sparse:
            x = x.to_dense()
        x_u, x_v = x[row], x[col]
        # 计算相似度得分
        simi_score = F.cosine_similarity(x_u, x_v)
        # 过滤低分边
        # mask = simi_score >= self.threshold
        # edge_index = edge_index[:, mask]
        # simi_score = simi_score[mask]
        # row, col = edge_index
        mask = simi_score < self.threshold
        simi_score[mask] = 10e-6
        simi_sum = torch_scatter.scatter_add(simi_score, col, dim_size=x.size(0))  # 每个节点与周围节点相似度总和
        simi_score_norm = simi_score / simi_sum[row]

        if add_self_loop:
            # 自环权重
            degrees = torch_scatter.scatter_add(torch.ones_like(simi_score_norm), col, dim_size=x.size(0))
            self_weight = float(1) / (degrees + 1)
            # 邻点权重
            simi_score_norm = (degrees[row] / (degrees[row] + 1)) * simi_score_norm
            simi_score_norm = torch.cat([simi_score_norm, self_weight])
            loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)

        simi_score_norm = simi_score_norm.exp()

        return edge_index, simi_score_norm

    def forward(self, x, edge_index):

        for i in range(self.nlayers):
            if i == 0:
                edge_index, edge_weight = self.get_attention_coefficient(x, edge_index, self.add_self_loops)
            else:
                edge_index, edge_weight = self.get_attention_coefficient(x, edge_index, False)
                if self.with_gate:
                    edge_weight = self.gate * edge_weight_memory + (1 - self.gate) * edge_weight
            edge_weight_memory = edge_weight.detach().clone()
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            if i == self.nlayers - 1:
                break
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