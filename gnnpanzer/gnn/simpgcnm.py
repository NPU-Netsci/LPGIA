import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

import utils

from typing import Union, Optional, Callable, Optional

from tqdm import tqdm
from torch.nn import Parameter
from torch_geometric.nn.conv import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product

from utils import Arrays


class SimPGCNModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Union[int, Arrays], num_layers: int, out_channels: int,
                 dropout: float = 0.5, with_bias=True, with_relu=True, with_softmax=True,
                 with_bn=False, with_ln=False,
                 normalize=True, add_self_loops=True, conv_cached=False,
                 knn_cached=True, knn_cached_path='./simpgcn/knn_cache/', gamma=0.1, ssl_lambda=5, bias_init=0):
        super().__init__()
        self.nlayers = num_layers
        # init conv
        # not use cache for evasion setting or inductive setting
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lns = nn.ModuleList()
        # normalize and add self loop 不通过gcnconv实现
        normalize = False
        add_self_loops = False
        conv_cached = False

        # hidden dim list
        if type(hidden_channels) == int:
            hidden_channels = (np.ones(num_layers-1) * hidden_channels).astype(int)
        else:
            assert len(hidden_channels) == num_layers-1, 'dim not match'
        # input
        self.convs.append(GCNConv(in_channels, hidden_channels[0],
                                  bias=with_bias, normalize=normalize, add_self_loops=add_self_loops, cached=conv_cached))
        if with_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels[0]))
        if with_ln:
            self.lns.append(nn.LayerNorm(in_channels))
        # middle_hidden
        for layer_i in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels[layer_i], hidden_channels[layer_i+1],
                                      bias=with_bias, normalize=normalize, add_self_loops=add_self_loops, cached=conv_cached))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels[layer_i+1]))
            if with_ln:
                self.lns.append(nn.LayerNorm(hidden_channels[layer_i]))
        # output
        if with_ln:
            self.lns.append(nn.LayerNorm(hidden_channels[-1]))
        self.convs.append(GCNConv(hidden_channels[-1], out_channels,
                                  bias=with_bias, normalize=normalize, add_self_loops=add_self_loops, cached=conv_cached))
        #
        self.scores = nn.ParameterList()
        self.scores.append(Parameter(torch.FloatTensor(in_channels, 1)))
        for i in range(len(hidden_channels)):
            self.scores.append(Parameter(torch.FloatTensor(hidden_channels[i], 1)))

        self.score_bias = nn.ParameterList()
        self.score_bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(len(hidden_channels)):
            self.score_bias.append(Parameter(torch.FloatTensor(1)))

        self.d_k = nn.ParameterList()
        self.d_k.append(Parameter(torch.FloatTensor(in_channels, 1)))
        for i in range(len(hidden_channels)):
            self.d_k.append(Parameter(torch.FloatTensor(hidden_channels[i], 1)))

        self.d_bias = nn.ParameterList()
        self.d_bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(len(hidden_channels)):
            self.d_bias.append(Parameter(torch.FloatTensor(1)))

        # discriminator for ssl
        self.ssl_conv = nn.Linear(hidden_channels[-1], 1)

        self.add_loss = 0

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
        #
        self.identity = None
        self.adj_knn = None
        self.knn_cached = knn_cached
        self.knn_cahce_path = knn_cached_path
        self.gamma = gamma
        self.ssl_lambda = ssl_lambda
        self.embedding = None
        self.pseudo_labels = None
        self.node_pairs = None
        self.sims = None
        self.bias_init = bias_init

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

        for s in self.scores:
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)
        for b in self.score_bias:
            # fill in b with postive value to make
            # score s closer to 1 at the beginning
            b.data.fill_(self.bias_init)

        for dk in self.d_k:
            stdv = 1. / math.sqrt(dk.size(1))
            dk.data.uniform_(-stdv, stdv)
        for b in self.d_bias:
            b.data.fill_(0)

    def forward(self, x, edge_index):
        # not support edge_weight and edge_attr yet
        device = edge_index.device
        adj = utils.transform_sparse_mat_type(edge_index, 'csr')
        self.identity = utils.transform_sparse_mat_type(sp.eye(adj.shape[0]).tocsr(), 'edge_index').to(device)
        # self.adj_knn = None
        if self.adj_knn is None:
            # adj knn -- no self loop norm tensor.sparse.floattensor
            # note: don't let tensor and numpy share memory
            self.adj_knn = self.get_knn_graph(x.detach().cpu().clone().numpy(),
                                              knn_cached=self.knn_cached, cache_path=self.knn_cahce_path).to(device)

        adj_knn = self.adj_knn
        knn_edge_index = utils.transform_sparse_mat_type(adj_knn, 'edge_index').to(device)
        knn_edge_weight = adj_knn.values().detach().to(device)

        adj_norm = utils.normalize_adj(adj, add_self_loop=True)
        edge_weight = torch.FloatTensor(adj_norm.tocoo().data).to(device)
        edge_index_norm = utils.transform_sparse_mat_type(adj_norm, 'edge_index').to(device)
        gamma = self.gamma
        x0 = x.detach().clone()

        for i in range(self.nlayers):
            if self.with_ln:
                x = self.lns[i](x)

            s = torch.sigmoid(x @ self.scores[i] + self.score_bias[i])
            dk = x @ self.d_k[i] + self.d_bias[i]

            x = s * self.convs[i](x, edge_index_norm, edge_weight) + (1-s) * self.convs[i](x, knn_edge_index, knn_edge_weight) + \
                gamma * dk * self.convs[i](x, self.identity)

            if i == self.nlayers - 2:
                self.embedding = x.clone()  # 不detach
                self.add_loss = self.ssl_lambda * self.regression_loss(x0, self.embedding)

            if i == self.nlayers - 1:
                break

            if self.with_bn:
                x = self.bns[i](x)

            # if self.with_relu:
            #     x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        if self.with_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

    def get_knn_graph(self, features, k=20, knn_cached=False, cache_path='./simpgcn/knn_cache/'):
        # create path
        if not os.path.exists(cache_path) and knn_cached:
            os.makedirs(cache_path)
        #
        if not knn_cached or not os.path.exists(cache_path + 'knn_graph_{}.npz'.format(features.shape)):
            features[features != 0] = 1
            sims = cosine_similarity(features)
            if knn_cached:
                np.save(cache_path + 'cosine_sims_{}.npy'.format(features.shape), sims)

            sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
            for i in range(len(sims)):
                indices_argsort = np.argsort(sims[i])
                sims[i, indices_argsort[: -k]] = 0

            adj_knn = sp.csr_matrix(sims)
            if knn_cached:
                sp.save_npz(cache_path + 'knn_graph_{}.npz'.format(features.shape), adj_knn)
        else:
            print('loading knn graph from: ' + cache_path + 'knn_graph_{}.npz'.format(features.shape))
            adj_knn = sp.load_npz(cache_path + 'knn_graph_{}.npz'.format(features.shape))

        adj_knn = utils.normalize_adj(adj_knn, add_self_loop=False)
        adj_knn = utils.transform_sparse_mat_type(adj_knn, 'tensor').float()

        return adj_knn

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

    def regression_loss(self, x, embeddings):
        if self.pseudo_labels is None:
            self.pseudo_labels = self.get_pseudo_labels(x.detach().cpu().clone().numpy(),
                                                        knn_cached=self.knn_cached, cache_path=self.knn_cahce_path).to(x.device)

        k = 10000
        node_pairs = self.node_pairs
        if len(self.node_pairs[0]) > k:
            sampled = np.random.choice(len(self.node_pairs[0]), k, replace=False)

            embeddings0 = embeddings[node_pairs[0][sampled]]
            embeddings1 = embeddings[node_pairs[1][sampled]]
            embeddings = self.ssl_conv(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels[sampled], reduction='mean')
        else:
            embeddings0 = embeddings[node_pairs[0]]
            embeddings1 = embeddings[node_pairs[1]]
            embeddings = self.ssl_conv(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels, reduction='mean')
        # print(loss)
        return loss

    def get_pseudo_labels(self, features, k=5, knn_cached=False, cache_path='./simpgcn/knn_cache/'):
        # features = features.detach().cpu().clone().numpy()
        features[features != 0] = 1
        if not os.path.exists(cache_path+'cosine_sims_{}.npy'.format(features.shape)):
            sims = cosine_similarity(features)
            if knn_cached:
                np.save(cache_path+'cosine_sims_{}.npy'.format(features.shape), sims)
        else:
            print('loading cos from: '+cache_path+'cosine_sims_{}.npy'.format(features.shape))
            sims = np.load(cache_path+'cosine_sims_{}.npy'.format(features.shape))

        if not os.path.exists(cache_path+'attrsim_sampled_idx_{}.npy'.format(features.shape)):
            try:
                indices_sorted = sims.argsort(1)
                idx = np.arange(k, sims.shape[0] - k)
                selected = np.hstack((indices_sorted[:, :k],
                                      indices_sorted[:, -k - 1:]))

                selected_set = set()
                for i in range(len(sims)):
                    for pair in product([i], selected[i]):
                        if pair[0] > pair[1]:
                            pair = (pair[1], pair[0])
                        if pair[0] == pair[1]:
                            continue
                        selected_set.add(pair)

            except MemoryError:
                selected_set = set()
                for ii, row in enumerate(sims):
                    row = row.argsort()
                    idx = np.arange(k, sims.shape[0] - k)
                    sampled = np.random.choice(idx, k, replace=False)
                    for node in np.hstack((row[:k], row[-k - 1:], row[sampled])):
                        if ii > node:
                            pair = (node, ii)
                        else:
                            pair = (ii, node)
                        selected_set.add(pair)

            sampled = np.array(list(selected_set)).transpose()
            if knn_cached:
                np.save(cache_path+'attrsim_sampled_idx_{}.npy'.format(features.shape), sampled)
        else:
            print('loading simi sample from: '+cache_path+'attrsim_sampled_idx_{}.npy'.format(features.shape))
            sampled = np.load(cache_path+'attrsim_sampled_idx_{}.npy'.format(features.shape))
        print('number of sampled:', len(sampled[0]))
        self.node_pairs = (sampled[0], sampled[1])
        self.sims = sims
        return torch.FloatTensor(sims[self.node_pairs]).reshape(-1, 1)


