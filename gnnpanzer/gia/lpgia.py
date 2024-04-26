import math
import time
import argparse
import random
import copy
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils

from torch.nn.parameter import Parameter
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.cluster import KMeans

from gnnpanzer.gnn import predict_gnn, test_gnn, LabelPropagation


class LPGIA(nn.Module):
    """
    基于标签传播的图注入算法, ver.2024/3/15
    """

    def __init__(self, surrogate_model, patience=200, n_top_k=10,
                 y_score_alpha=0.5, bwc_score_alpha=0.5, bp_score_alpha=-0.5):
        """"""
        super(LPGIA, self).__init__()
        # model info
        self.surrogate_model = surrogate_model
        self.device = 'cpu'
        # graph info
        self.nnodes = 0  #
        self.nclass = 0  #
        self.nfeats = 0  #
        self.ori_adj = None  # scipy sparse matrix
        self.ori_features = None  # np
        # attack res
        self.modified_adj = None  # scipy sparse matrix
        self.modified_features = None  # np
        # model special property
        # statistical info
        self.average_feature_num = 1
        self.avg_label_features = None
        self.prob_label_features = None
        self.mean_degree = 1
        self.patience = patience
        self.current_patience = 0
        self.n_top_k = n_top_k
        self.y_score_alpha = y_score_alpha
        self.bwc_score_alpha = bwc_score_alpha
        self.bp_score_alpha = bp_score_alpha
        self.nodes_bwc = None

    def get_label_features(self, labels, features, idx=None):
        if type(labels) == torch.Tensor:
            labels = labels.cpu().numpy()
        # if type(features) != sp.csr_matrix:
        #     features = sp.csr_matrix(features)
        # if idx is None:
        #     idx = np.arange(len(labels))
        if idx is not None:
            labels = labels[idx]
            features = features[idx]
        label_range = np.arange(self.nclass)
        avg_label_features = np.zeros((self.nclass, self.nfeats))  # 平均取值 sum(value)/sum(nonzero)
        avg_label_features_abs = np.zeros((self.nclass, self.nfeats))
        zero_label_features = np.zeros((self.nclass, self.nfeats))
        prob_label_features = np.zeros((self.nclass, self.nfeats))  # 出现概率 sum(nonzero)/nnodes
        max_label_features = np.zeros((self.nclass, self.nfeats))
        min_label_features = np.zeros((self.nclass, self.nfeats))
        # 综合考虑离散特征和连续特征
        for tag in label_range:
            label_feature = features[labels == tag]
            binary_label_feature = copy.deepcopy(label_feature)
            binary_label_feature[binary_label_feature != 0] = 1
            feature_count = binary_label_feature.sum(0)
            feature_count[feature_count == 0] = 1
            avg_label_feature = label_feature.sum(0) / feature_count
            prob_label_feature = np.mean(binary_label_feature, axis=0)
            zero_feature = (prob_label_feature == 0)
            avg_label_features[tag, :] = avg_label_feature
            avg_label_features_abs[tag, :] = np.abs(label_feature).sum(0) / feature_count
            zero_label_features[tag, :] = zero_feature
            prob_label_features[tag, :] = prob_label_feature
            max_label_features[tag, :] = label_feature.max(axis=0)
            min_label_features[tag, :] = label_feature.min(axis=0)

        self.avg_label_features = avg_label_features
        self.avg_label_features_abs = avg_label_features_abs
        self.prob_label_features = prob_label_features
        self.max_label_features = max_label_features
        self.min_label_features = min_label_features

    @torch.no_grad()
    def get_single_feature_pred(self, adj, features, batch=None):
        x_single = sp.diags(np.ones(self.nfeats))
        x = sp.vstack((sp.csr_matrix(features), x_single))
        x = utils.transform_sparse_mat_type(x, 'tensor')
        adj_ts = utils.transform_sparse_mat_type(adj, 'tensor')
        row = adj_ts._indices()[0]
        col = adj_ts._indices()[1]
        edge_index = torch.vstack((row, col))
        # output = self.surrogate_model.predict(x=x, edge_index=edge_index, device=self.device) 
        output = predict_gnn(self.surrogate_model, x, edge_index, self.device) 
        weight = torch.exp(output)[self.nnodes:, :]
        return weight

    @torch.no_grad()
    def get_predict_labels(self, adj, features):
        #
        # output = self.victim.predict(features=features, adj=adj)
        # pyg
        output = self.surrogate_model.predict(x=features, edge_index=adj, device=self.device)
        return output.argmax(1), output

    @torch.no_grad()
    def get_linear_weight(self):
        """获取线性化的权重矩阵，探测模型中的二维矩阵并相乘"""
        weight = None
        for para in self.surrogate_model.parameters():
            if para.ndim == 1:
                continue
            if weight is None:
                weight = para.detach()
            else:
                weight = weight @ para.detach()
        return weight

    def get_best_wrong_class(self, output: torch.Tensor, use_true_labels=False, labels=None):
        """获得最佳误导分类，即概率第二大标签"""
        bwc = []
        for i in range(output.shape[0]):
            tmp = output[i, :].detach().clone()
            if use_true_labels:
                pred_class = labels[i]
            else:
                pred_class = tmp.argmax().item()
            tmp[pred_class] -= 10000
            best_wrong_class = tmp.argmax().item()
            bwc.append(best_wrong_class)
        return torch.LongTensor(bwc)

    def get_fake_node_degrees(self, adj, n_fake_nodes, n_victim_nodes=None, strategy='sample'):

        degrees = np.array(adj.sum(1)).flatten()
        degree_mean = int(degrees.mean())
        print('degree_mean:', degree_mean)
        if strategy == 'fix':
            fake_node_degrees = np.repeat(degree_mean, n_fake_nodes)  # 定值
        elif strategy == 'sample':
            sample_degrees = degrees.copy()
            sample_mean = degree_mean + 1
            sample_degrees[sample_degrees >= sample_mean * 2] = sample_mean * 2
            fake_node_degrees = random.sample(sample_degrees.tolist(), n_fake_nodes)  # 为注入节点采样度
            fake_node_degrees = np.array(fake_node_degrees, dtype=int)
        elif strategy == 'direct':
            if n_victim_nodes is None:
                raise ValueError()
            fake_node_degrees = np.repeat(int(n_victim_nodes/n_fake_nodes), n_fake_nodes)
        print('fix victim:', degree_mean * n_fake_nodes, ' || sample victim:', np.sum(fake_node_degrees))
        # 增删统一限制
        if n_victim_nodes is None:
            n_victim_nodes = degree_mean * n_fake_nodes
        margin = int(fake_node_degrees.sum() - n_victim_nodes)
        if margin > 0:
            for i in range(margin):
                idx = np.random.choice(np.arange(len(fake_node_degrees))[fake_node_degrees != 1],
                                       1, replace=False)
                fake_node_degrees[idx] -= 1
        else:
            for i in range(-margin):
                idx = np.random.choice(np.arange(len(fake_node_degrees))[fake_node_degrees != fake_node_degrees.max()],
                                       1, replace=False)
                fake_node_degrees[idx] += 1
        print('final victim:', np.sum(fake_node_degrees), ' || final fake:', n_fake_nodes)
        return fake_node_degrees  # np

    def get_node_homophily_scores(self, adj, labels, nodes=None, mode='decrease'):
        """计算节点同质性得分"""
        # labels = labels.to('cpu')
        if nodes is None:
            nodes = np.arange(adj.shape[0])
        node_homophily_scores = np.zeros_like(nodes, dtype=float)
        for i, node in enumerate(nodes):
            neig = adj[node].indices  # adj is sparse m

            if mode == 'bp':
                s = (labels[1][node] == labels[0][neig]).sum()
            else:
                s = (labels[node] == labels[neig]).sum()

            if mode == 'decrease':
                # 得分为添加注入节点后同质性降低程度 s/t - s/(t+1) = s/(t^2+t)
                node_homophily_scores[i] = s / (len(neig) * len(neig) + len(neig))
            elif mode == 'normal':
                node_homophily_scores[i] = s / len(neig)
            elif mode == 'bp':
                node_homophily_scores[i] = s / len(neig)
            else:
                raise ValueError()
        node_sorted_idx = np.argsort(-node_homophily_scores)
        victim_node_idx = nodes[node_sorted_idx]

        return node_homophily_scores, victim_node_idx

    def attack(self, ori_adj, ori_features, labels, idx_train, idx_val, idx_test,
               n_fake_nodes, n_victim_nodes=None, n_inject_nodes=None, victim_nodes=None,
               device='cpu', verbose=False, label_access='query',
               opt_fake_feature='weight', opt_homophily_score='hs', opt_cluster='lp',
               bwc_cached=None, surrogate_out_cached=None, weight_cached=None,
               sample_strategy='sample', feature_budget=None, sparse_feature=True):
        """

        """
        # 初始信息获取
        self.device = device
        self.nnodes = ori_adj.shape[0]  # csr
        self.nclass = int(labels.max() + 1)
        self.nfeats = ori_features.shape[1]  # 2d np
        self.ori_adj = ori_adj.tolil(copy=True)
        self.ori_features = ori_features.copy()
        if type(labels) != torch.Tensor:
            labels = torch.LongTensor(labels)
        labels = labels.to(device)
        ori_degrees = np.array(ori_adj.sum(1)).flatten()
        self.modified_adj = ori_adj.copy()
        self.modified_features = ori_features.copy()
        modified_adj = ori_adj.copy()
        modified_features = ori_features.copy()

        if victim_nodes is None:
            victim_nodes = idx_test
        # 注入节点idx总列表
        total_fake_nodes = np.array(range(self.nnodes, self.nnodes + n_fake_nodes), dtype=int)
        # feature-attack budget
        if sparse_feature:
            self.average_feature_num = int(np.diff(sp.csr_matrix(ori_features).indptr).mean())  # 根据每行第一个元素所在位置计算每行特征数
        else:
            self.average_feature_num = ori_features.shape[1]
        # 采样节点度并分配注入节点度
        fake_node_degrees = self.get_fake_node_degrees(ori_adj, n_fake_nodes, n_victim_nodes, sample_strategy)
        # n_top_k = 10

        # 提取替代模型线性化权重矩阵
        if weight_cached is None:
            surrogate_weight = self.surrogate_model.get_linear_weight().to(device)
        else:
            surrogate_weight = weight_cached.to(device)
        # pyg格式转换
        # x = utils.sparse_mx_to_torch_sparse_tensor(ori_features)
        x = torch.FloatTensor(ori_features.copy()).to(device)
        y = labels.clone()
        # adj_ts = utils.sparse_mx_to_torch_sparse_tensor(ori_adj)
        adj_ts = utils.transform_sparse_mat_type(ori_adj, 'tensor')
        row = adj_ts._indices()[0]
        col = adj_ts._indices()[1]
        edge_index = torch.vstack((row, col))
        # 获取节点分类概率矩阵
        if surrogate_out_cached is None:
            # surrogate_out = self.surrogate_model.predict(x, edge_index, device)
            surrogate_out = predict_gnn(self.surrogate_model, x, edge_index, self.device)
            surrogate_out_soft = torch.exp(surrogate_out)
        else:
            surrogate_out = torch.log(surrogate_out_cached)
            surrogate_out_soft = surrogate_out_cached
        # 初始化伪标签
        if label_access == 'query':
            predict_labels = surrogate_out_soft.argmax(dim=1)
            predict_labels[idx_train] = labels[idx_train]
            predict_labels[idx_val] = labels[idx_val]
        elif label_access == 'truth':
            predict_labels = labels
        else:
            raise NotImplementedError('label access match failed')
        # 获取特征统计
        # self.get_label_features(predict_labels, ori_features, idx_train)
        self.get_label_features(predict_labels, ori_features)
        if opt_fake_feature == 'query_weight':
            query_weight = self.get_single_feature_pred(ori_adj, ori_features)
        # 获取节点最优误导分类
        if bwc_cached is None:
            if label_access == 'query':
                nodes_bwc = self.get_best_wrong_class(surrogate_out_soft)
            elif label_access == 'truth':
                nodes_bwc = self.get_best_wrong_class(surrogate_out_soft, True, labels)
            else:
                raise ValueError()
        else:
            nodes_bwc = bwc_cached
        self.nodes_bwc = nodes_bwc.cpu().numpy()
        # 按最优扰动分类对受害节点分组
        victim_bwc = np.hstack((victim_nodes.reshape(-1, 1),
                                nodes_bwc[victim_nodes].detach().cpu().numpy().reshape(-1, 1)))
        victim_node_groups = []
        group_volume = np.zeros(self.nclass)
        for i in range(self.nclass):
            victim_node_groups.append(victim_bwc[victim_bwc[:, 1] == i, 0])
            group_volume[i] = len(victim_node_groups[i])
        # 获取节点预测标签同质性下降得分
        victim_de_homo_y_scores, rank_de_homo_y_victim = self.get_node_homophily_scores(ori_adj, predict_labels,
                                                                                        victim_nodes, 'decrease')
        # 获取节点最佳扰动标签同质性得分
        victim_homo_bwc_scores, rank_homo_bwc_victim = self.get_node_homophily_scores(ori_adj, nodes_bwc,
                                                                                      victim_nodes, 'normal')
        # 获取节点最佳扰动标签与邻居预测标签同质性得分
        victim_homo_bp_scores, rank_homo_bwc_victim = self.get_node_homophily_scores(ori_adj,
                                                                                     [predict_labels.to('cpu'), nodes_bwc.to('cpu')],
                                                                                     victim_nodes, 'bp')
        if opt_homophily_score == 'hs':
            victim_scores = victim_de_homo_y_scores * self.y_score_alpha + victim_homo_bwc_scores * self.bwc_score_alpha + (
                            victim_homo_bp_scores * self.bp_score_alpha)
        elif opt_homophily_score == 'hd':
            victim_scores = victim_de_homo_y_scores
        elif opt_homophily_score == 'dp':
            alpha_d = 0.33
            weight1 = 0.9
            weight2 = 0.1
            victim_scores = torch.mul(surrogate_out_soft[np.arange(self.nnodes), predict_labels], alpha_d) + (1 - alpha_d)
            victim_scores = weight1*victim_scores/ori_degrees + weight2*victim_scores/np.sqrt(ori_degrees)
            victim_scores = victim_scores[victim_nodes].numpy()
        elif opt_homophily_score == 'random':
            victim_scores = np.random.rand(len(victim_nodes))
        else:
            raise NotImplementedError()
        victim_scores = np.hstack((victim_nodes.reshape(-1, 1), victim_scores.reshape(-1, 1)))
        # 受害节点注入限制
        single_node_victim_budget = {}
        for v in victim_nodes:
            single_node_victim_budget.update({int(v): int(ori_degrees[v])})
        # 顺序注入
        # 批量注入预处理
        n_injected_nodes = 0  # 已注入节点数
        # if n_inject_nodes is None:
        #     n_inject_nodes = n_fake_nodes - n_injected_nodes  # 无设定则一次完成注入
        n_inject_nodes = 1  # 目前只支持一次注入一个节点
        n_victimed_nodes = 0
        # prepare for calculate
        if x.is_sparse:
            x = x.to_dense()
        surrogate_out_soft = surrogate_out_soft.detach().cpu().numpy()
        ori_surrogate_out = surrogate_out_soft.copy()
        victim_memory = []
        wait_victim_scores = victim_scores[np.argsort(-victim_scores[:, 1]), :]
        while n_injected_nodes < n_fake_nodes:
            #
            modified_degrees = np.array(modified_adj.sum(1)).flatten()
            # 生成候选方案
            # 选择当前得分最高的节点作为初始节点
            init_node_idx = np.array([wait_victim_scores[0, 0]], dtype=int)
            target_class = nodes_bwc[init_node_idx].item()
            # 复制候选分组
            potential_nodes = victim_node_groups[target_class].copy()
            potential_nodes = np.delete(potential_nodes, np.where(potential_nodes == init_node_idx))
            # 迭代选择下一个节点，该节点与已有候选点应当尽可能提高注入节点被分类为目标分类的概率
            cluster_node = [init_node_idx]
            cluster_out = surrogate_out_soft[cluster_node] * np.power(modified_degrees[cluster_node]+1, -0.5).reshape(-1, 1)
            # cluster_out = cluster_out.reshape(1, -1)
            # 个例：注入预算超过了分组中剩余节点数
            cluster_budget = fake_node_degrees[n_injected_nodes]
            if cluster_budget > len(potential_nodes):
                cluster_budget = len(potential_nodes)
            for i in range(1, cluster_budget):
                # 生成候选解
                potential_cluster = np.hstack((np.repeat(cluster_node, len(potential_nodes), axis=0),
                                               potential_nodes.reshape(-1, 1)))
                # 使用lp算法扩散概率
                potential_out = np.repeat(cluster_out, len(potential_nodes), axis=0) + (
                                surrogate_out_soft[potential_nodes] * np.power(
                                modified_degrees[potential_nodes] + 1, -0.5).reshape(-1, 1))
                if opt_cluster == 'lp':
                    # scores = potential_out[:, target_class] / potential_out.sum(axis=1)
                    scores = potential_out[:, target_class] - potential_out[np.arange(potential_out.shape[0]), potential_out.argmax(axis=1)]
                    if scores.shape[0] > self.n_top_k:
                        potential_cluster_id = np.argsort(-scores)[:self.n_top_k]
                    else:
                        potential_cluster_id = np.arange(scores.shape[0])
                    #
                    # potential_cluster_id = np.random.choice(potential_cluster_id, 1, replace=False)
                    # lp算法作为初筛条件，取同质性得分最高节点作为局部最优解
                    potential_new_nodes = potential_cluster[potential_cluster_id, -1]
                    node_scores = victim_scores[np.where(victim_scores[:, 0] == potential_new_nodes[:, None])[1], 1]
                    best_node_idx = node_scores.argmax(axis=0)  # 相对位置
                    potential_cluster_id = [potential_cluster_id[best_node_idx]]  # 取高分对于簇id

                elif opt_cluster == 'tk':
                    potential_new_nodes = potential_cluster[:, -1]
                    node_scores = victim_scores[np.where(victim_scores[:, 0] == potential_new_nodes[:, None])[1], 1]
                    best_node_idx = node_scores.argmax(axis=0)  # 绝对位置
                    potential_cluster_id = [best_node_idx]  # 取高分对于簇id
                elif opt_cluster == 'random':
                    best_node_idx = np.random.choice(np.arange(potential_cluster.shape[0]), 1, replace=False)[0]  # 绝对位置
                    potential_cluster_id = [best_node_idx]  # 取高分对于簇id
                else:
                    raise NotImplementedError(opt_cluster)

                # 更新聚类
                cluster_node = potential_cluster[potential_cluster_id]
                cluster_out = potential_out[potential_cluster_id]
                potential_nodes = np.delete(potential_nodes, np.where(potential_nodes == cluster_node[0, -1]))

            # 为注入节点生成特征
            # 按特征标签映射矩阵得分选取
            if opt_fake_feature == 'weight':
                cluster_class = int(cluster_out.argmax(axis=1))
                if cluster_class != target_class:
                    feature_scores = surrogate_weight[:, target_class] - surrogate_weight[:, cluster_class]
                else:
                    feature_scores = surrogate_weight[:, target_class]
                feature_scores = torch.mul(feature_scores, torch.FloatTensor(self.avg_label_features[target_class, :]).to(device))
                # feature_scores = surrogate_weight[:, target_class]
                top_feature_idx = torch.argsort(-feature_scores).cpu().numpy()
            # 按统计概率选取
            elif opt_fake_feature == 'statistics' or opt_fake_feature == 'dense' or opt_fake_feature == 'clone':
                top_feature_idx = np.argsort(-self.prob_label_features[target_class, :])
            elif opt_fake_feature == 'query_weight':
                cluster_class = int(cluster_out.argmax(axis=1))
                if cluster_class != target_class:
                    feature_scores = query_weight[:, target_class] - query_weight[:, cluster_class]
                else:
                    feature_scores = query_weight[:, target_class]
                feature_scores = torch.mul(feature_scores, torch.FloatTensor(self.avg_label_features[target_class, :]))
                top_feature_idx = torch.argsort(-feature_scores).cpu().numpy()
            elif opt_fake_feature == 'random':
                top_feature_idx = np.arange(self.nfeats)
                np.random.shuffle(top_feature_idx)
            elif opt_fake_feature == 'wsf':
                cluster_class = int(cluster_out.argmax(axis=1))
                if cluster_class != target_class:
                    feature_scores = surrogate_weight[:, target_class] - surrogate_weight[:, cluster_class]
                else:
                    feature_scores = surrogate_weight[:, target_class]
                feature_scores = torch.mul(feature_scores, torch.FloatTensor(self.avg_label_features[target_class, :]).to(device))
                feature_scores = torch.mul(feature_scores, torch.FloatTensor(self.prob_label_features[target_class, :]).to(device))
                # feature_scores = surrogate_weight[:, target_class]
                top_feature_idx = torch.argsort(-feature_scores).cpu().numpy()
            else:
                raise NotImplementedError()

            fake_feature = np.zeros(self.nfeats)
            # 增加多样性，中毒场景表现更好
            if feature_budget is None:
                top_feature_idx = np.random.choice(top_feature_idx[:self.average_feature_num*2], self.average_feature_num, replace=False)
            elif feature_budget == -1:
                top_feature_idx = feature_scores > 0
            else:
                num = torch.sum(feature_scores > 0)
                top_feature_idx = np.random.choice(top_feature_idx[:num], int(num*feature_budget), replace=False)
            # 直接最高得分，逃避场景表现更好
            # top_feature_idx = top_feature_idx[:self.average_feature_num]
            fake_feature[top_feature_idx] = self.avg_label_features[target_class, top_feature_idx]
            # dense abs when have negative
            if opt_fake_feature == 'dense':
                cluster_class = int(cluster_out.argmax(axis=1))
                if cluster_class != target_class:
                    feature_scores = surrogate_weight[:, target_class] - surrogate_weight[:, cluster_class]
                else:
                    feature_scores = surrogate_weight[:, target_class]
                feature_scores = feature_scores.detach().cpu().numpy()
                # dense approximate average
                fake_feature[feature_scores>0] = self.max_label_features[target_class, feature_scores>0] / 2
                fake_feature[feature_scores<0] = self.min_label_features[target_class, feature_scores<0] / 2
            # clone
            if opt_fake_feature == 'clone':
                z = sp.csr_matrix(self.ori_features) @ sp.csr_matrix(surrogate_weight.cpu().numpy())
                # softmax?
                z = utils.transform_sparse_mat_type(z, 'tensor').to_dense()
                # z = torch.softmax(z, dim=1)
                cluster_class = int(cluster_out.argmax(axis=1))
                if cluster_class != target_class:
                    nf_scores = z[:, target_class] - z[:, cluster_class]
                else:
                    nf_scores = z[:, target_class]
                nf = copy.deepcopy(self.ori_features)
                # best_nf = torch.argmax(nf_scores).numpy()
                top_nf = torch.argsort(-nf_scores).cpu().numpy()
                top_nf = top_nf[nf_scores[top_nf] > nf_scores[top_nf[0]]* 0.9]
                best_nf = np.random.choice(top_nf, 1, replace=False)
                fake_feature = nf[best_nf]

            # 正式注入
            if self.current_patience > 0:
                print('inject node:', n_injected_nodes, '/ total node:', n_fake_nodes, '/ patience', self.current_patience)
            old_patience = self.current_patience
            self.current_patience = 0

            # 候选方案生成完毕，注入图中
            # modified_features = sp.vstack((modified_features, fake_feature)).tocsr()
            modified_features = np.vstack((modified_features, fake_feature))
            modified_adj = utils.reshape_csr(modified_adj.tocsr(),
                                             shape=(modified_adj.shape[0] + 1, modified_adj.shape[1] + 1))
            modified_adj = modified_adj.tolil()
            for v in cluster_node[0]:
                victim_memory.append(v)
                edge = (v, total_fake_nodes[n_injected_nodes])
                modified_adj[tuple(edge)] = modified_adj[tuple(edge[::-1])] = 1 - modified_adj[tuple(edge)]
            modified_adj = modified_adj.tocsr()
            modified_adj.eliminate_zeros()

            # 将已受害点移除候选集
            # group_volume[target_class] -= fake_node_degrees[n_injected_nodes]
            for v in cluster_node[0]:
                wait_victim_scores = np.delete(wait_victim_scores, np.where(wait_victim_scores[:, 0] == v), axis=0)
                v_class = nodes_bwc[v]
                group_volume[v_class] -= 1
                victim_node_groups[v_class] = np.delete(victim_node_groups[v_class],
                                                        np.where(victim_node_groups[v_class] == v))

            # 更新surrogate_output
            x = torch.FloatTensor(modified_features)
            adj_ts = utils.transform_sparse_mat_type(modified_adj, 'tensor')
            row = adj_ts._indices()[0]
            col = adj_ts._indices()[1]
            edge_index = torch.vstack((row, col))
            surrogate_out_new = predict_gnn(self.surrogate_model, x, edge_index, device)
            surrogate_out_soft_new = torch.exp(surrogate_out_new).detach()
            surrogate_out = surrogate_out_new
            lp_model = LabelPropagation(num_layers=50, alpha=0.9,
                                        memory=False, add_self_loops=True, auto_convergence=False)
            lp_output = lp_model(surrogate_out_soft_new.to(device), edge_index.to(device))
            lp_output = (lp_output / lp_output.sum(dim=1).reshape(-1, 1)).detach().cpu().numpy()
            surrogate_out_soft = lp_output

            n_injected_nodes += 1

            # if not tqdm
            if n_injected_nodes % 100 == 0:
                print('has been injected:', n_injected_nodes, 'nodes')

        self.modified_adj = modified_adj.tocsr()
        self.modified_features = modified_features

        return self.modified_adj, self.modified_features

